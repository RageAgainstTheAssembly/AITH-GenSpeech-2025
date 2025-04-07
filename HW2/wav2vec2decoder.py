from typing import List, Tuple
import heapq
import kenlm
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


class Wav2Vec2Decoder:
    def __init__(
            self,
            model_name="facebook/wav2vec2-base-960h",
            lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
            beam_width=3,
            alpha=1.0,
            beta=1.0
        ):
        """
        Initialization of Wav2Vec2Decoder class
        
        Args:
            model_name (str): Pretrained Wav2Vec2 model from transformers
            lm_model_path (str): Path to the KenLM n-gram model (for LM rescoring)
            beam_width (int): Number of hypotheses to keep in beam search
            alpha (float): LM weight for shallow fusion and rescoring
            beta (float): Word bonus for shallow fusion
        """
        # once logits are available, no other interactions with the model are allowed
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)

      


        # you can interact with these parameters
        self.vocab = {i: c for c, i in self.processor.tokenizer.get_vocab().items()}
        self.blank_token_id = self.processor.tokenizer.pad_token_id
        self.word_delimiter = self.processor.tokenizer.word_delimiter_token
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.lm_model = kenlm.Model(lm_model_path) if lm_model_path else None

    def greedy_decode(self, logits: torch.Tensor) -> str:
        """
        Perform greedy decoding (find best CTC path)
        
        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V)
        
        Returns:
            str: Decoded transcript
        """
        log_probs = torch.log_softmax(logits, dim=-1)

        pred_ids = torch.argmax(log_probs, dim=-1).tolist()

        previous_id = -1
        decoded_ids = []

        for token_id in pred_ids:
            if token_id != self.blank_token_id and token_id != previous_id:
                decoded_ids.append(token_id)
            previous_id = token_id

        decoded_chars = [self.vocab[id] for id in decoded_ids]

        transcript = ''.join(decoded_chars).replace(self.word_delimiter, ' ')
    
        return transcript

    def beam_search_decode(self, logits: torch.Tensor, return_beams: bool = False):
      """
      Perform beam search decoding (no LM)
      
      Args:
          logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
              T - number of time steps and
              V - vocabulary size
          return_beams (bool): Return all beam hypotheses for second pass LM rescoring
      
      Returns:
          Union[str, List[Tuple[float, List[int]]]]: 
              (str) - If return_beams is False, returns the best decoded transcript as a string.
              (List[Tuple[List[int], float]]) - If return_beams is True, returns a list of tuples
                  containing hypotheses and log probabilities.
      """
      import heapq

      log_probs = torch.log_softmax(logits, dim=-1)

      beam = [(0.0, [], [])]

      for t in range(log_probs.size(0)):
          new_beam = []

          for neg_log_p, token_seq, time_aligned_seq in beam:
              for v in range(log_probs.size(1)):
                  token_log_p = log_probs[t, v].item()
                  new_neg_log_p = neg_log_p - token_log_p

                  if v == self.blank_token_id:
                      new_beam.append((new_neg_log_p, token_seq, time_aligned_seq))
                  else:
                      is_repeated_from_consecutive_frame = (
                          time_aligned_seq and 
                          time_aligned_seq[-1][0] == v and 
                          time_aligned_seq[-1][1] == t - 1
                      )
                      
                      if is_repeated_from_consecutive_frame:
                          new_time_aligned = time_aligned_seq[:-1] + [(v, t)]
                          new_beam.append((new_neg_log_p, token_seq, new_time_aligned))
                      else:
                          new_token_seq = token_seq + [v]
                          new_time_aligned = time_aligned_seq + [(v, t)]
                          new_beam.append((new_neg_log_p, new_token_seq, new_time_aligned))

          beam = heapq.nsmallest(self.beam_width, new_beam, key=lambda x: x[0])

      beams = [(token_seq, -neg_log_p) for neg_log_p, token_seq, _ in beam]
      best_tokens, best_score = max(beams, key=lambda x: x[1])
      decoded_chars = [self.vocab[id] for id in best_tokens]
      best_hypothesis = ''.join(decoded_chars).replace(self.word_delimiter, ' ')
      
      if return_beams:
          return beams
      else:
          return best_hypothesis
        
    

    def beam_search_with_lm(self, logits: torch.Tensor) -> str:
      """
      Perform beam search decoding with shallow LM fusion
      
      Args:
          logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
              T - number of time steps and
              V - vocabulary size
      
      Returns:
          str: Decoded transcript
      """
      if not self.lm_model:
          raise ValueError("KenLM model required for LM shallow fusion")
      
      import heapq

      log_probs = torch.log_softmax(logits, dim=-1)
      
      beam = [(0.0, 0.0, [], None, -1, "", "")]

      for t in range(log_probs.size(0)):
          new_beam = []

          for neg_score, acoustic_log_p, tokens, last_token_id, last_time, text, partial_word in beam:
              for v in range(log_probs.size(1)):
                  token_log_p = log_probs[t, v].item()
                  new_acoustic_log_p = acoustic_log_p + token_log_p

                  if v == self.blank_token_id:
                      new_beam.append((neg_score - token_log_p, new_acoustic_log_p, 
                                      tokens, last_token_id, last_time, text, partial_word))

                  elif v == last_token_id and t == last_time + 1:
                      new_beam.append((neg_score - token_log_p, new_acoustic_log_p,
                                      tokens, v, t, text, partial_word))
                  
                  else:
                      new_tokens = tokens + [v]

                      char = self.vocab[v]
                      new_text = text
                      new_partial = partial_word

                      lm_score = 0.0
                      if char == self.word_delimiter:
                          if new_partial:
                              word_with_space = new_partial + " "
                              lm_score = self.lm_model.score(new_text + word_with_space) - self.lm_model.score(new_text)
                              new_text += word_with_space
                              new_partial = ""
                      else:
                          new_partial += char

                      word_count = len(new_text.strip().split())

                      combined_score = new_acoustic_log_p + self.alpha * lm_score + self.beta * word_count

                      new_beam.append((-combined_score, new_acoustic_log_p, new_tokens, v, t, new_text, new_partial))

          beam = heapq.nsmallest(self.beam_width, new_beam, key=lambda x: x[0])

      final_beam = []
      for neg_score, acoustic_log_p, tokens, _, _, text, partial_word in beam:
          final_lm_score = 0.0

          if partial_word:
              final_text_with_word = text + partial_word
              final_lm_score = self.lm_model.score(final_text_with_word) - self.lm_model.score(text)

          final_score = (-neg_score) + self.alpha * final_lm_score
          final_beam.append((-final_score, tokens, partial_word))

      _, best_tokens, final_partial = min(final_beam, key=lambda x: x[0])

      decoded_chars = [self.vocab[id] for id in best_tokens]

      best_hypothesis = ''.join(decoded_chars).replace(self.word_delimiter, ' ')
      if final_partial:
          best_hypothesis += final_partial
      
      return best_hypothesis.strip()



    def lm_rescore(self, beams: List[Tuple[List[int], float]]) -> str:
      """
      Perform second-pass LM rescoring on beam search outputs
      
      Args:
          beams (list): List of tuples (hypothesis, log_prob)
      
      Returns:
          str: Best rescored transcript
      """
      if not self.lm_model:
          raise ValueError("KenLM model required for LM rescoring")
      
      best_score = float('-inf')
      best_transcript = ""
      
      for tokens, acoustic_log_p in beams:
          if not tokens:
              continue

          chars = [self.vocab[token] for token in tokens]
          text = ''.join(chars).replace(self.word_delimiter, ' ').strip()

          if not text:
              continue

          lm_score = self.lm_model.score(text, bos=True, eos=False)

          word_count = len(text.split())

          combined_score = acoustic_log_p + self.alpha * lm_score + self.beta * word_count

          if combined_score > best_score:
              best_score = combined_score
              best_transcript = text

      if not best_transcript:
          return ""
          
      return best_transcript


    def decode(self, audio_input: torch.Tensor, method: str = "greedy") -> str:
        """
        Decode input audio file using the specified method
        
        Args:
            audio_input (torch.Tensor): Audio tensor
            method (str): Decoding method ("greedy", "beam", "beam_lm", "beam_lm_rescore"),
                where "greedy" is a greedy decoding,
                      "beam" is beam search without LM,
                      "beam_lm" is beam search with LM shallow fusion, and 
                      "beam_lm_rescore" is a beam search with second pass LM rescoring
        
        Returns:
            str: Decoded transcription
        """
        inputs = self.processor(audio_input, return_tensors="pt", sampling_rate=16000)
        #print(inputs)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(inputs['input_values'].squeeze(0)).logits[0]

        if method == "greedy":
            return self.greedy_decode(logits)
        elif method == "beam":
            return self.beam_search_decode(logits)
        elif method == "beam_lm":
            return self.beam_search_with_lm(logits)
        elif method == "beam_lm_rescore":
            beams = self.beam_search_decode(logits, return_beams=True)
            return self.lm_rescore(beams)
        else:
            raise ValueError("Invalid decoding method. Choose one of 'greedy', 'beam', 'beam_lm', 'beam_lm_rescore'.")


def test(decoder, audio_path, true_transcription):

    import Levenshtein

    audio_input, sr = torchaudio.load(audio_path)
    assert sr == 16000, "Audio sample rate must be 16kHz"

    print("=" * 60)
    print("Target transcription")
    print(true_transcription)

    # Print all decoding methods results
    for d_strategy in ["greedy", "beam", "beam_lm", "beam_lm_rescore"]:
        print("-" * 60)
        print(f"{d_strategy} decoding") 
        transcript = decoder.decode(audio_input, method=d_strategy)
        print(f"{transcript}")
        print(f"Character-level Levenshtein distance: {Levenshtein.distance(true_transcription, transcript.strip())}")


if __name__ == "__main__":
    
    test_samples = [
        ("examples/sample1.wav", "IF YOU ARE GENEROUS HERE IS A FITTING OPPORTUNITY FOR THE EXERCISE OF YOUR MAGNANIMITY IF YOU ARE PROUD HERE AM I YOUR RIVAL READY TO ACKNOWLEDGE MYSELF YOUR DEBTOR FOR AN ACT OF THE MOST NOBLE FORBEARANCE"),
        ("examples/sample2.wav", "AND IF ANY OF THE OTHER COPS HAD PRIVATE RACKETS OF THEIR OWN IZZY WAS UNDOUBTEDLY THE MAN TO FIND IT OUT AND USE THE INFORMATION WITH A BEAT SUCH AS THAT EVEN GOING HALVES AND WITH ALL THE GRAFT TO THE UPPER BRACKETS HE'D STILL BE ABLE TO MAKE HIS PILE IN A MATTER OF MONTHS"),
        ("examples/sample3.wav", "GUESS A MAN GETS USED TO ANYTHING HELL MAYBE I CAN HIRE SOME BUMS TO SIT AROUND AND WHOOP IT UP WHEN THE SHIPS COME IN AND BILL THIS AS A REAL OLD MARTIAN DEN OF SIN"),
        ("examples/sample4.wav", "IT WAS A TUNE THEY HAD ALL HEARD HUNDREDS OF TIMES SO THERE WAS NO DIFFICULTY IN TURNING OUT A PASSABLE IMITATION OF IT TO THE IMPROVISED STRAINS OF I DIDN'T WANT TO DO IT THE PRISONER STRODE FORTH TO FREEDOM"),
        ("examples/sample5.wav", "MARGUERITE TIRED OUT WITH THIS LONG CONFESSION THREW HERSELF BACK ON THE SOFA AND TO STIFLE A SLIGHT COUGH PUT UP HER HANDKERCHIEF TO HER LIPS AND FROM THAT TO HER EYES"),
        ("examples/sample6.wav", "AT THIS TIME ALL PARTICIPANTS ARE IN A LISTEN ONLY MODE"),
        ("examples/sample7.wav", "THE INCREASE WAS MAINLY ATTRIBUTABLE TO THE NET INCREASE IN THE AVERAGE SIZE OF OUR FLEETS"),
        ("examples/sample8.wav", "OPERATING SURPLUS IS A NON CAP FINANCIAL MEASURE WHICH IS DEFINED AS FULLY IN OUR PRESS RELEASE"),
    ]

    decoder = Wav2Vec2Decoder()

    _ = [test(decoder, audio_path, target) for audio_path, target in test_samples]

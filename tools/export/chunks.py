import sentencepiece, sys
from pyrt import to_model_stress, split_chunks, prepare_text_prompt
sp = sentencepiece.SentencePieceProcessor("/assets/weights/tokenizer.model")
text = sys.argv[1]
stressed = to_model_stress(text)
print("stressed:", stressed)
for i, c in enumerate(split_chunks(sp, stressed, 50, False, False)):
    prepared, guess = prepare_text_prompt(c, False, False)
    print(f"chunk {i} frames_after_eos={guess+2} prepared={prepared}")
    print("  ids=[" + ",".join(map(str, sp.encode(prepared))) + "]")

def split_sentences(text):
  sentences = []
  start = 0
  for i,ch in enumerate(text):
     if ch in '.?!' and i+1 < len(text) and text[i+1] == ' ':
        word = text[start:i].split(' ')[-1]
        if '.' in word or (len(word) == 1 and word.isupper()):
           continue
        sentences.append(text[start:i+1].strip())
        start = i+1
  leftover = text[start:].strip()
  return sentences, leftover
tests = [
    ("The sky is blue. It scatters light.", (["The sky is blue."], "It scatters light.")),
    ("Washington D.C. is the capital. Next", (["Washington D.C. is the capital."], "Next")),
    ("It costs 3.5 dollars. Ok", (["It costs 3.5 dollars."], "Ok")),
    ("Hello", ([], "Hello")),
]

for text, expected in tests:
    result = split_sentences(text)
    print('PASS' if result == expected else f'FAIL: {text!r} -> {result!r}')
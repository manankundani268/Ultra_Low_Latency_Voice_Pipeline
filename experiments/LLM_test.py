from ollama import chat
import time

start = time.perf_counter()

instruction = 'Answer in one or two short sentences, as if speaking aloud.'

stream = chat(
  model='llama3.2:1b',
  messages=[{'role': 'system', 'content': instruction},
            {'role': 'user', 'content': 'Why is the sky blue?'}],
  stream=True,
)

pieces = []
first = None
for chunk in stream:
  if first is None:
     first = (time.perf_counter() - start) * 1000
  pieces.append(chunk.message.content)
  print(chunk.message.content, end='', flush=True)

total = (time.perf_counter() - start)*1000

print()
reply = ''.join(pieces)
print(f'Reply length : {len(reply)} characters')

print(f'first TTFT : {first:.1f}')
print(f'Total time : {total:.1f}')
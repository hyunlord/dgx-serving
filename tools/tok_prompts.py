from transformers import AutoTokenizer
import numpy as np, sys
tok=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-27B')
SP=sys.argv[1]
prompts={
 "code":"# Python: a class with getters and setters for many fields.\nclass Person:\n    def __init__(self, name, age, email, address, phone, city, country, zipcode):\n",
 "fact":"List of country capitals.\nFrance: Paris\nJapan: Tokyo\nGermany: Berlin\nItaly: Rome\nSpain: Madrid\nCanada:",
 "prose":"Once upon a time, in a quiet village nestled between two great mountains, there lived an old clockmaker who",
}
for k,p in prompts.items():
    ids=np.array(tok(p).input_ids,dtype=np.int32)
    ids.tofile(f"{SP}/p_{k}.bin")
    print(f"{k}: {len(ids)} tokens")

"""
01_build_5gram_kenlm.py
Builds a 5-gram KenLM from RegSpeech12 transcripts (5x upweighted).
Requires KenLM CLI compiled at /content/kenlm_src/build/bin/.

Build KenLM CLI first (one-time):
    apt-get install -y build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
    git clone https://github.com/kpu/kenlm.git /content/kenlm_src
    cd /content/kenlm_src && mkdir -p build && cd build && cmake .. && make -j 4
"""
import os
KENLM = "/content/kenlm_src/build/bin"
corpus = "kenlm_training_corpus.txt"
os.system(KENLM + "/lmplz -o 5 --discount_fallback < " + corpus + " > lm_5gram.arpa")
os.system(KENLM + "/build_binary lm_5gram.arpa lm_5gram.bin")
print("Built lm_5gram.bin")

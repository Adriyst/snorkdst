BASE_PATH="/usr/local/Development"
source "${BASE_PATH}/snorkdst/venv/bin/activate"
python "${BASE_PATH}/snorkdst/gen_snorkel_set.py"
mv "${BASE_PATH}/snorkdst/dstc2_snorkel_en.json" "${BASE_PATH}/master/data/clean/"

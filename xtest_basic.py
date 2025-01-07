import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_NAME = "jbochi/madlad400-3b-mt"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

# Test input
#text = "<2en> I love pizza!"
text = "<2en> V skladu z zahtevami programa MD-11 Program samovrednotenja in postopka ADP-1.0.011 Izvajanje internih samovrednotenj v organizacijskih enotah, je bilo v času od 15. junija do 15. julija 2024 izvedeno samovrednotenje izvajanja procesa dedikacije komercialnih proizvodov (CGD)."
input_ids = tokenizer(text, return_tensors="pt", max_length=128, truncation=True).input_ids.to(device)

# Generate output
outputs = model.generate(input_ids, max_length=128)
translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Translation: {translation}")



import torch
torch.cuda.is_available()
#True
torch.cuda.device_count()
# 1
torch.cuda.device(0)
# <torch.cuda.device object at 0x000002506F83A1A0>
torch.cuda.get_device_name(0)
# 'NVIDIA GeForce RTX 4090'
print(torch.__version__)
# 2.5.1
print(torch.tensor([1.0, 2.0]).cuda())
# tensor([1., 2.], device='cuda:0')
x = torch.rand(5, 3)
print(x)
Model: [AdamCodd/vit-base-nsfw-detector](https://https//huggingface.co/AdamCodd/vit-base-nsfw-detector/tree/main)

Or using model on local by manualy download at [here](https://huggingface.co/yeftakun/vit-base-nsfw-detector/tree/main):
- config.json
- preprocessor_config.json
- model.safetensors

```
# Direct
processor = ViTImageProcessor.from_pretrained('yeftakun/vit-base-nsfw-detector')
model = AutoModelForImageClassification.from_pretrained('yeftakun/vit-base-nsfw-detector')

# Local
processor = ViTImageProcessor.from_pretrained('./')
model = AutoModelForImageClassification.from_pretrained('./')
```

Then run `app.py`

from transformers import DonutProcessor, VisionEncoderDecoderModel, VisionEncoderDecoderConfig
import utils.helpers as helpers

def model_loader(dataset, max_length, pretrained_model): 
    width, height = helpers.image_size(dataset)
    image_size = [height, width]

    config = VisionEncoderDecoderConfig.from_pretrained(pretrained_model)
    config.encoder.image_size = image_size
    config.decoder.max_length = max_length

    processor = DonutProcessor.from_pretrained(pretrained_model)
    model = VisionEncoderDecoderModel.from_pretrained(pretrained_model, config=config)

    return processor, model


# HeR-T: Herbarium specimen label Recognition Transformer  

![License](https://img.shields.io/github/license/elderprince/HeR-T-Fine-tuning.git)

## 🚀 Overview  
HeR-T (Herbarium specimen label Recognition Transformer) is a fine-tuned vision-language model designed for automated metadata extraction of history specimen labels, especially herbarium specimen labels. It leverages Donut-base and has been fine-tuned with 55,089 herbarium specimen images from the Herbarium of the University of Pisa (international acronym PI). 

## 🔥 Features  
- **Fine-tuned on** specimen images from the Herbarium of the University of Pisa for automated metadata extraction of history specimen labels
- **Supports** image inputs with labels containing printed, handwritten, or mixed-format texts  
- **Evaluation**: Tree Edit Distance (TED) accuracy score with the formula max(0, 1−TED(pr, gt)/TED(φ, gt)), where gt, pr, and φ stand for ground truth, prediction, and empty trees respectively 
- **Pre-trained weights** are loaded from Donut-base on Hugging Face(naver-clova-ix/donut-base)

## 🏗️ Installation  
```bash
conda create -n HeR-T python=3.9
conda activate HeR-T
git clone https://github.com/elderprince/HeR-T-Fine-tuning.git
cd HeR-T-Fine-tuning
pip install -r requirements.txt
```

## 💻 Usage  
### Inference  
```python
from your_model_package import Model

model = Model.load("path/to/weights")
result = model.predict("path/to/image.jpg")
print(result)
```

### Fine-tuning  
```bash
python train.py --config config.yaml
```

## 📊 Performance  
| Metric | Value |
|--------|-------|
| Accuracy | XX% |
| F1 Score | XX% |
| Inference Time | XX ms |

## 📂 Model Checkpoints  
Download pre-trained weights from [link to model checkpoints].  

## 📜 Dataset  
[Explain the dataset used for fine-tuning, linking to sources if publicly available.]  

## 🏆 Results  
[Add qualitative results, sample images, and predictions.]  

## 🤝 Contributing  
We welcome contributions! Please check the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.  

## 📜 License  
This project is licensed under the [License Name] License - see the [LICENSE](LICENSE) file for details.  

## 🔗 References  
- [Paper/Blog Post related to this model]  
- [Original base model]  

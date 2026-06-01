# NLP-project
# Projet ASR — Reconnaissance Automatique de la Parole Amazigh

## Description
Ce projet consiste à fine-tuner un modèle de reconnaissance vocale sur des données Amazigh (Tamazight / Kabyle). Il combine 5 datasets HuggingFace, normalise les transcriptions en arabizi, complète les traductions manquantes via NLLB-200, puis fine-tune wav2vec2-large-xlsr-53-arabic. Les performances sont mesurées avec le WER (Word Error Rate) avant et après fine-tuning.

## Objectifs
- Harmoniser 5 datasets Amazigh (scripts arabe, tifinagh, latin)
- Compléter les traductions manquantes via NLLB-200
- Évaluer les modèles Whisper et Wav2Vec2 en baseline
- Fine-tuner wav2vec2-large-xlsr-53-arabic sur les données Amazigh
- Mesurer l'amélioration du WER avant/après fine-tuning

## Technologies utilisées
- Python, PyTorch
- HuggingFace (transformers, datasets, evaluate)
- Modèles : wav2vec2-large-xlsr-53-arabic, facebook/nllb-200-distilled-600M, Whisper
- Pandas, NumPy, scikit-learn, jiwer
- Google Colab (GPU T4)

## Structure du projet
- `projet_asr.py` : script principal (Google Colab)
- `dataset_final.csv` : dataset compilé toutes sources
- `train.csv / val.csv / test.csv` : splits 80/10/10
- `wav2vec2-amazigh-final/` : modèle fine-tuné sauvegardé
- `README.md` : documentation du projet

## Résultats
- Baseline WER (avant fine-tuning) : ~102 %
- WER après fine-tuning : ~96 %
- Pas d'overfitting détecté (train loss et val loss évoluent de manière similaire)
- Limitation : le dataset tamawalt-n-imZZyann a été exclu (erreur SSL sur le serveur audio tal2.ircam.ma)

## Auteur
Rayan Rachedi et Myriam El Atmioui — M2 MSI Data Science, IAE Orléans — 2025-2026

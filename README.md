# Fake_News_Detector
## Problem Statement
 Fake news spreads quickly across digital platforms, and single
feature detection models often fail when news appears in
 multiple forms, such as short text, long articles, or images. This
 project aims to solve this problem by developing a multi-modal
 fake news detection system that combines text, article, and
 image-based analysis for higher accuracy and robustness.

## PROPOSED SOLUTION
 The proposed solution is to develop a multi-modal fake news detection system that
 analyzes text, full articles, and images within a unified framework. The approach
 involves collecting and preprocessing datasets for all three formats, applying text
 cleaning for written content, and using Tesseract OCR to extract text from images.
 TF-IDF vectorization converts all processed text into numerical features, which are
 then used to train Logistic Regression and Random Forest models. These models are
 evaluated using accuracy, precision, recall, and confusion matrix metrics. By
 integrating text-based, article-based, and image-assisted detection into one
 pipeline, the system provides a more accurate and reliable method for identifying
 fake news across diverse formats.

## GOALS AND OBJECTIVES
 - Build a multi-modal fake news detection system (Text + Article + Image)
 - Categories Train and evaluate ML models (Logistic Regression & Random Forest) for high accuracy.
 - Integrate OCR based image analysis to detect misinformation in visual content.

## TECHNOLOGY USED
 - Python 
 - Logistic Regression and Random Forest Classifier
 - TF-IDF Vectorization
 - Tesseract OCR
 - NumPy & Pandas

## KEY FEATURES OF FAKE NEWS DETECCTOR
- Uses OCR to detect text inside images
- Extracts meaningful features from text using TF-IDF
- Works across short text, long articles, and images
- More reliable and robust than single-input models
- Modular and scalable pipeline for future improvements

## Future Prospects
- Extend the system to support multiple languages
- Integrate with social media platforms for real-time scanning
- Deploy for news agencies to automatically verify incoming content

import src.Utils as utils

"""
Loads a trained & stored LDA Model 
explore documents based on topic information 
"""

model_path="..\\..\\models\\saved_lda_model.pkl"
lda_model, tfidf,tfidf_vectorizer,data_content=utils.load_topic_model(model_path)


utils.top_documents_per_topic(lda_model, tfidf, data_content, num_topics=5, number_of_documents=50, topic_number=3)
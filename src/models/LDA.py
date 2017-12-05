from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation


class LDA:
    """
       Parameters
       ----------
       n_topics : int
           Number of topics
       n_iter : int, default 2000
           Number of sampling iterations
       alpha : float, default 0.1
           Dirichlet parameter for distribution over topics
       eta : float, default 0.01
           Dirichlet parameter for distribution over words
       random_state : int or RandomState, optional
           The generator used for the initial topics.


       Feature-related parameters:

       max_df: float, default 0.95,

       min_df=2,

       max_features
       """

    def __init__(self, n_topics=10, n_iter=2000, alpha=0.1, eta=0.01,
                 max_df=0.95, min_df=2, max_features=None):

        self.n_topics = n_topics
        self.n_iter = n_iter
        self.alpha = alpha
        self.eta = eta


        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features

    def fit_model_TFIDF(self, data_samples):
        """
        trains LDA model on the data_samples
        returns: the trained LDA_model, the vectorizer (configuration) and features (tfidf)
        """
        tfidf_vectorizer = TfidfVectorizer(max_df=self.max_df,
                                           min_df=self.min_df,
                                           max_features=self.max_features,
                                           stop_words='english')

        tfidf = tfidf_vectorizer.fit_transform(data_samples)

        lda_model = LatentDirichletAllocation(n_components=self.n_topics,
                                              max_iter=self.n_iter,
                                              learning_method='online',
                                              learning_offset=10.,
                                              random_state=0)

        lda_model.fit(tfidf)

        return lda_model, tfidf_vectorizer, tfidf

    def print_top_words(self, lda_model, feature_names, n_top_words):
        '''Prints the n top words from the model.

        :param model: The model
        :param feature_names: The feature names
        :param n_top_words: Number of top words
        '''
        store_topics = []
        for topic_idx, topic in enumerate(lda_model.components_):
            print("Topic #%d:" % topic_idx)
            topic_keywords = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
            print(topic_keywords)
            # store_topics.append(topic_keywords)
        print()
        # Utils.write_list_to_file("topic_keywords.txt", store_topics)





import src.Utils as utils
import src.models.LDA as lda_module


def get_parser():
    """Get parser object for script xy.py."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,

                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("-n_topics", default=10, type=int,
                        help="Number of topics. Default 10")

    parser.add_argument("-n_iterations", default=10, type=int,
                        help="Number of Iterations. Default 10")

    parser.add_argument("-n_features", default=1000, type=int,
                        help="Maximum Number of Features. Default 100")

    parser.add_argument("-max_df", default=0.85, type=float,
                        help="Maximum Document Frequency. Removes frequent words. "
                             "Default 0.85( Words that appear in more than 85% of documents are removed) ")

    parser.add_argument("-min_df", default=5, type=int,
                        help="Minimum Document Frequency. Removes Infrequent words. "
                             "Default 5( Words that appear in less than 5 documents are removed) ")

    parser.add_argument("-store", default=True, type=bool,
                        help="Store the trained model for later use")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    # Read raw dataset
    docPaths = ["..\\..\\data\\raw\\binary_positive.txt"]  # ,"DataModule/binary_negative.txt"]
    res = utils.loadTextDocuments(docPaths)
    data = res["data"]
    data_content = [data[i]["content"] for i in range(len(data))]
    data_labels = [data[i]["class"] for i in range(len(data))]

    #Create an Instance of LDA
    lda_instance = lda_module.LDA(n_topics=args.n_topics, max_df=args.max_df, min_df=args.min_df,
                                  n_iter=args.n_iterations,
                                  max_features=args.n_features)

    #Train the Model
    lda_model, tfidf_vectorizer, tfidf = lda_instance.fit_model_TFIDF(data_content)

    #Store the model if specified by the input argument
    if args.store:
        utils.save_topic_model([lda_model, tfidf, tfidf_vectorizer,data_content],"..\\..\\models\\saved_lda_model.pkl")

    #Print Topics
    tf_feature_names = tfidf_vectorizer.get_feature_names()
    lda_instance.print_top_words(lda_model, tf_feature_names, n_top_words=10)
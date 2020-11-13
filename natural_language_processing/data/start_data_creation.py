

def menu():
    from natural_language_processing.data.text_processing import TextProcessing
    choice = input("What data do you want to include: internal only, press 1, internal and external, press 12: ")
    if choice == 1:
        print("Creation of internal data file.")
        data_proc = TextProcessing()
        data_proc.process_train_data()
        x_train, y_train, x_val, y_val = data_proc.split_data()
        data_proc.store_h5py(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
    else:
        try:
            print("Creation of data files of internal  processed data.")
            data_proc = TextProcessing()
            data_proc.process_train_data()
            x_train, y_train, x_val, y_val = data_proc.split_data()
            print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
            data_proc.store_h5py(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)

            from natural_language_processing.data.parse_word_embeddings import ParseWordEmbeddings
            word_index = data_proc.indexing_informs_tokenizer()
            embeddings_matrix = ParseWordEmbeddings.create_embeddings_matrix(word_index)  # the pretrainned weights of NN
            print("Creation of data files of external  processed data.")
            ParseWordEmbeddings.store_h5py(embeddings_matrix)

        except RuntimeError:
            raise Exception("Probably run out of enough memory for the processing and storing.")


menu()


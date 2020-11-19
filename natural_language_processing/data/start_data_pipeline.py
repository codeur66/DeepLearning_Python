

def menu():
    from natural_language_processing.data.text_processing import TextProcessing
    try:
        choice = input("What data do you want to process and store: "
                       "internal only, press: 1, "
                       "internal and pretrained: 2, "
                       "exit menu: 3 ")
        while choice != "3":

            if choice == "1":
                print("Creation of data and files from in house text processing.")
                data_proc = TextProcessing()
                data_proc.process_train_data()
                x_train, y_train, x_val, y_val = data_proc.split_data()
                data_proc.store_h5py(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
            elif choice == "2":
                print("Process data and store files from in house text processing and pretrained model.")
                data_proc = TextProcessing()
                data_proc.process_train_data()
                x_train, y_train, x_val, y_val = data_proc.split_data()
                print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
                data_proc.store_h5py(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)

                from natural_language_processing.data.parse_word_embeddings import ParseWordEmbeddings
                word_index = data_proc.indexing_informs_tokenizer()
                embeddings_matrix = ParseWordEmbeddings.create_embeddings_matrix(word_index)  # the pretrainned weights of NN
                ParseWordEmbeddings.store_h5py(embeddings_matrix)
            elif choice == 3:
                exit(0)
            else:
                print('Unavailable choice.')
                choice = input("What data do you want to process: "
                               "internal only, press: 1, "
                               "internal and pretrained:2, "
                               "exit menu: 3 ")
    except RuntimeError:
        raise Exception("Probably run out of enough memory for the data processing and storing.")


menu()


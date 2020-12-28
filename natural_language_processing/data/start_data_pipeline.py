from natural_language_processing.configurations.configuration_infrastructure import Config
from natural_language_processing.configurations.configurations import CFG
from natural_language_processing.data.text_processing import TextProcessing

from natural_language_processing.logging.LoggerCls import LoggerCls
import os.path


def menu():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    formatter = '%(name)s - %(levelname)s  - %(message)s'
    logToFile = LoggerCls("log_to_file", "start_data_piepeline:", dir_path + "/data_piepeline.log", "w", formatter,
                        "INFO")
    logStream = LoggerCls("log_to_stdout", "start_data_pipeline: ", None, "w", formatter, "INFO")

    try:
        config = Config.from_json(CFG)
        choice = input("What data do you want to process and store: "
                       "internal only, press: 1, "
                       "internal and pretrained: 2, "
                       "exit menu: 3 \n ")
        while choice != "3":

            if choice == "1":
                logToFile.info("Creation of data and files from in house text processing.")
                logStream.info("Creation of data and files from in house text processing.")
                # print("Creation of data and files from in house text processing.")
                data_proc = TextProcessing()
                data_proc.process_train_tst_data(config.data.path_test_data, "train")
                x_train, y_train, x_val, y_val = data_proc.split_data()

                data_proc.process_train_tst_data(config.data.path_test_data, "test")
                x_test, y_test = data_proc.shape_tensors_and_store_data()
                data_proc.store_h5py(x_train=x_train, y_train=y_train,
                                     x_val=x_val, y_val=y_val,
                                     x_test=x_test, y_test=y_test)

                break
            elif choice == "2":
                logToFile.info("Process data and store files from in house text processing and pretrained model.")
                logStream.info("Process data and store files from in house text processing and pretrained model.")
                # print("Process data and store files from in house text processing and pretrained model.")
                data_proc = TextProcessing()
                data_proc.process_train_tst_data(config.data.path_test_data, "train")
                x_train, y_train, x_val, y_val = data_proc.split_data()

                data_proc.process_train_tst_data(config.data.path_test_data, "test")
                x_test, y_test = data_proc.shape_tensors_and_store_data()
                data_proc.store_h5py(x_train=x_train, y_train=y_train,
                                     x_val=x_val, y_val=y_val,
                                     x_test=x_test, y_test=y_test)

                from natural_language_processing.data.parse_word_embeddings import ParseWordEmbeddings
                word_index = data_proc.indexing_informs_tokenizer()
                embeddings_matrix = ParseWordEmbeddings.create_embeddings_matrix(word_index)  # the pretrainned weights of NN
                ParseWordEmbeddings.store_h5py(embeddings_matrix)

                break
            elif choice == 3:
                exit(0)
            else:
                logStream.warning("\n Unavailable choice.")
                # print("Unavailable choice.")
                choice = input("What data do you want to process: "
                               "internal only, press: 1, "
                               "internal and pretrained:2, "
                               "exit menu: 3 \n")
    except (TypeError, AttributeError, RuntimeError) as e:
        logToFile.error("Probably run out of enough memory for the data processing and storing.")
        logToFile.error(e)
        raise Exception("Probably run out of enough memory for the data processing and storing.")


menu()

import csv

train_fileHeader = ["local_model", "round", "epoch", "internal_epoch", "average_loss", "accuracy", "correct_data",
                    "total_data"]
test_fileHeader = ["model", "epoch", "average_loss", "accuracy"]
train_result = []  # train_fileHeader
test_result = []  # test_fileHeader
posiontest_result = []  # test_fileHeader

dp_fileHeader = ["epoch",  "eps", "acc", "clean_loss","adv_loss","acc_p"]
dp_result =[]

def clear_csv():
    if len(train_result)>0:
        train_result.clear()
    if len(test_result)>0:
        test_result.clear()
    if len(posiontest_result)>0:
        posiontest_result.clear()
    if len(dp_result)>0:
        dp_result.clear()


def save_result_csv(folder_path,run_idx=0):
    if len(train_result)>0:
        train_csvFile = open(f'{folder_path}/train_result_{run_idx}.csv', "w")
        train_writer = csv.writer(train_csvFile)
        train_writer.writerow(train_fileHeader)
        train_writer.writerows(train_result)
        train_csvFile.close()

    if len(test_result)>0:
        test_csvFile = open(f'{folder_path}/test_result_{run_idx}.csv', "w")
        test_writer = csv.writer(test_csvFile)
        test_writer.writerow(test_fileHeader)
        test_writer.writerows(test_result)
        test_csvFile.close()

    if len(dp_result)>0:
        dp_csvFile=  open(f'{folder_path}/dp_result_{run_idx}.csv', "w")
        dp_writer = csv.writer(dp_csvFile)
        dp_writer.writerow(dp_fileHeader)
        dp_writer.writerows(dp_result)
        dp_csvFile.close()

    if len(posiontest_result)>0:
        test_csvFile = open(f'{folder_path}/posiontest_result_{run_idx}.csv', "w")
        test_writer = csv.writer(test_csvFile)
        test_writer.writerow(test_fileHeader)
        test_writer.writerows(posiontest_result)
        test_csvFile.close()

  
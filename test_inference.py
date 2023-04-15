from espnet2.bin.asr_inference import Speech2Text
import soundfile
import csv

speech2text = Speech2Text("exp/asr_train_asr_conformer_raw_zh_char_sp/config.yaml", "exp/asr_train_asr_conformer_raw_zh_char_sp/valid.acc.ave_10best.pth", device="cuda")

with open('test_output_conformer_80epoch_4head_300dev.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["id", "text"])
    for i in range(1,347):
        audio, rate = soundfile.read("downloads/data_aishell/wav/test/global/{}.wav".format(i))
        print(i,speech2text(audio)[0][0])
        writer.writerow([str(i), speech2text(audio)[0][0]])
        



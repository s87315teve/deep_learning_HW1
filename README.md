# Deep learning kaggle 01


程式要有說明，報告要包含使用資料，作法，模型，訓練過程，結果與檢討。

訓練步驟如下
* **step0: 前置作業**
使用git下載espnet
```
git clone https://github.com/espnet/espnet.git
```
建置環境(訓練的時候要記得把環境切到espnet)
```
cd espnet/tools
./setup_anaconda.sh anaconda espnet 3.9
make
```

到[kaggle](https://www.kaggle.com/competitions/espnet-taiwanese-asr1/overview)下載資料集，並把以下檔案改名
```
train -> train_origin (資料夾)
test -> test_origin (資料夾)
train-toneless.csv -> train-toneless_origin.csv
```



* **step1: 把聲音採樣率從22K變16K**
執行audio_transfer.py
```
python audio_transfer.py
```
使用for loop+subprocess在程式中自動使用sox的指令將train_origin與test_origin的裡面所有的音檔自動轉換為16KHz，並輸出為train、test資料夾
```python=
for filename in file_list:
    subprocess.getstatusoutput("sox {}/{} -r 16000 -e signed-integer -b 16 {}/{}".format(origin_folder, filename, new_folder, filename))
```
* **step2:資料清洗**
執行data_clean.py
```
python data_clean.py
```
使用正則表示式來對train-toneless_origin.csv進行資料清洗，將檔案中所有非英文字母以及非space的字元刪除(pattern的寫法如下)，最後再輸出成train.csv
```python=
pattern = re.compile("[^A-Za-z\s]")
```

* **step3: 檔案改名&把檔案放到指定位置**

把train.csv設定成用空白分割，並把檔案改名成aishell_transcript.txt，最後面再放上test_編號＋ a e i o u(如下圖)
![](https://i.imgur.com/qYpfdmV.png)

將各檔案移動到指定位置
(train為訓練資料集，dev為validation的資料)

```
train/1~300.wav -> egs2/aishell/asr1/downloads/data_aishell/wav/dev/1~300.wav
train/301~.wav -> egs2/aishell/asr1/downloads/data_aishell/wav/train/300~.wav
test/-> egs2/aishell/asr1/downloads/data_aishell/wav/test/
aishell_transcript.txt -> egs2/aishell/asr1/downloads/data_aishell/transcript/aishell_transcript.txt 

```
* **step4: 修改各種config檔**
    * run.sh裏面ngpu改成1
    * egs2/aishell/asr1/local/data.sh裏面 
        * 把download的地方註解掉
        * 設定訓練音檔路徑
        * 把處理空白的地方註解掉
    ```bash=
    # line 59~65
    # echo local/download_and_untar.sh ${download_opt} "${AISHELL}" "${data_url}" data_aishell
    # local/download_and_untar.sh ${download_opt} "${AISHELL}" "${data_url}" data_aishell
    # echo local/download_and_untar.sh ${download_opt} "${AISHELL}" "${data_url}" resource_aishell
    # local/download_and_untar.sh ${download_opt} "${AISHELL}" "${data_url}" resource_aishell

    aishell_audio_dir=${AISHELL}/data_aishell/wav
    aishell_text=${AISHELL}/data_aishell/transcript/aishell_transcript.txt

    # line 112~119
    # remove space in text
    #for x in train dev test; do
    #  cp data/${x}/text data/${x}/text.org
    #  paste -d " " <(cut -f 1 -d" " data/${x}/text.org) <(cut -f 2- -d" " data/${x}/text.org | tr -d " ") \
    #      > data/${x}/text
    #  rm data/${x}/text.org
    #done

    ```
    * run.sh改asr_config換模型  (optional)
    * 改conf/train_asr_\[modelname\].yaml檔裡面的attention_heads、batch_bins   (optional)

* **step5: 開始訓練**
```
run.sh
```
* **step6: 使用模型**
```
python test_inference.py
```
執行此程式即可對指定的音檔進行預測，能夠在程式中Speech2Text()的地方放入模型的config檔以及訓練好的模型，在本次的作業中主要是以前十名表現好的模型進行平均後輸出的結果，推測此模型的表現將會較穩定。
```python=
speech2text = Speech2Text("exp/asr_train_asr_conformer_raw_zh_char_sp/config.yaml", "exp/asr_train_asr_conformer_raw_zh_char_sp/valid.acc.ave_10best.pth", device="cuda")
```
---
## 調參數與model結果

在本次kaggle競賽中，嘗試了以下參數來訓練模型(結果圖在最下面)
| case | 模型 | 參數 | train:dev | epoch |
| --- | --- | --- | --- | --- |
| case1 | train_asr_branchformer| encoder attention_heads: 8 <br>decoder attention_heads: 8 | ~ : 200 | 100 |
| case2 | train_asr_conmformer | encoder attention_heads: 8 <br>decoder attention_heads: 8 | ~ : 200 | 100 |
| case3 | train_asr_conmformer | encoder attention_heads: 8 <br>decoder attention_heads: 8 | ~ : 200 | 60 |
| case4 | train_asr_conmformer | encoder attention_heads: 8 <br>decoder attention_heads: 8 | ~ : 300 | 80 |
| case5 | train_asr_conmformer | encoder attention_heads: 4 <br>decoder attention_heads: 4 | ~ : 300 | 80 |

* **結果與討論**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;首先從epoch的數量來看，能夠發現epoch大約在60~80之間，acc就已經收斂得差不多了，因此不需要訓練太多epoch；訓練資料集裡總共有約3000筆資料，在觀察train和valid的acc時，能夠發現train和dev資料的比例約為9:1時的表現較好，若dev的資料再減少的話則會導致train的acc高但valid的acc降低，也就是會造成over fitting的問題；此外在決定branchformer和conformer的選擇以及attention_heads的數量時，發現影響皆不太顯著，而layer數量若增加的話則會大幅增加對記憶體的需求，因此在本作業中並未多做調整。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在經過考慮後，最後交出的模型是case4的模型，雖然public data的準確度不是最高的，但他的epoch數量以及train:dev的數量皆在合適的範圍，因此選擇這個模型交出。

---
* case1
![](https://i.imgur.com/Vh1sbwe.png)
* case2
![](https://i.imgur.com/WyiL4gs.png)
* case3
![](https://i.imgur.com/8koMYRT.png)
* case4
![](https://i.imgur.com/OjnKHiu.png)
* case5
![](https://i.imgur.com/vm2kccr.png)



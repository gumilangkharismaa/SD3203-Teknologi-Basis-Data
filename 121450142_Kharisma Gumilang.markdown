---
jupyter:
  colab:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.10.7
  nbformat: 4
  nbformat_minor: 5
---

::: {#529lgqLglXVo .cell .markdown id="529lgqLglXVo"}
**Nama : Kharisma Gumilang**
**NIM : 121450142**
**Kelas : RB**
:::

::: {#YMwIni7g_8Z5 .cell .code id="YMwIni7g_8Z5"}
``` python
from google.colab import drive
drive.mount('/content/drive')
```
:::

::: {#ABhDSWOPemmp .cell .markdown id="ABhDSWOPemmp"}
Dalam artikel tersebut, disebutkan bahwa terdapat tiga metode utama
untuk menyimpan dan mengakses gambar menggunakan Python, yaitu melalui
format PNG, lightning memory-mapped databases (LMDB), dan hierarchical
data format (HDF5). Penelitian dilakukan dengan melakukan
langkah-langkah eksperimental, mulai dari persiapan, penyimpanan satu
gambar, penyimpanan banyak gambar, pembacaan satu gambar, pembacaan
banyak gambar, hingga evaluasi. Fokus utama dari artikel ini adalah
membandingkan proses penyimpanan dan pengaksesan gambar menggunakan
ketiga metode tersebut.
:::

::: {#s6FPTkZxenjq .cell .markdown id="s6FPTkZxenjq"}
Dalam langkah persiapan, dataset gambar yang digunakan berasal dari
Canadian Institute for Advanced Research (CIFAR-10), yang terdiri dari
60.000 gambar berwarna dengan ukuran 32x32 piksel.
:::

::: {#a7b919d3 .cell .markdown id="a7b919d3"}
# Tiga Cara Menyimpan dan Mengakses Banyak Gambar dalam Python
:::

::: {#f7983f5f .cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="f7983f5f" outputId="bae17cc9-07cf-4c68-d43f-69a1f38c6b75"}
``` python
import numpy as np
import pickle
from pathlib import Path

# Path to the unzipped CIFAR data
data_dir = Path("/content/drive/MyDrive/cifar-10-batches-py")

# Unpickle function provided by the CIFAR hosts
def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict

images, labels = [], []
for batch in data_dir.glob("data_batch_*"):
    batch_data = unpickle(batch)
    for i, flat_im in enumerate(batch_data[b"data"]):
        im_channels = []
        # Each image is flattened, with channels in order of R, G, B
        for j in range(3):
            im_channels.append(
                flat_im[j * 1024 : (j + 1) * 1024].reshape((32, 32))
            )
        # Reconstruct the original image
        images.append(np.dstack((im_channels)))
        # Save the label
        labels.append(batch_data[b"labels"][i])

print("Loaded CIFAR-10 training set:")
print(f" - np.shape(images)     {np.shape(images)}")
print(f" - np.shape(labels)     {np.shape(labels)}")
```

::: {.output .stream .stdout}
    Loaded CIFAR-10 training set:
     - np.shape(images)     (50000, 32, 32, 3)
     - np.shape(labels)     (50000,)
:::
:::

::: {#EfBJvvCLfMnc .cell .markdown id="EfBJvvCLfMnc"}
**Analisis :**
Penggunaan library seperti PIL memungkinkan pemrosesan dan pemodelan
data gambar, bahkan dalam jumlah besar seperti ratusan gambar. Namun,
menghadapi volume data yang besar, seperti dalam pelatihan model
Convolutional Neural Networks (CNN), menimbulkan tantangan dalam
efisiensi algoritma. Memuat data besar ke dalam memori untuk pelatihan
dapat memakan waktu yang signifikan dan kurang efisien, terutama saat
proses pemrosesan dilakukan dalam batch.
:::

::: {#98f17435 .cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="98f17435" outputId="78b2e2b0-5f8b-482c-cc36-5b6404f013a3"}
``` python
pip install Pillow #menginstallpillow untuk image manipulation
```

::: {.output .stream .stdout}
    Requirement already satisfied: Pillow in /usr/local/lib/python3.10/dist-packages (9.4.0)
:::
:::

::: {#fe418248 .cell .markdown id="fe418248"}
## Start With LMDB ( Lightning Memory - Mapped Database) {#start-with-lmdb--lightning-memory---mapped-database}
:::

::: {#BEv6ZqI4fyJ8 .cell .markdown id="BEv6ZqI4fyJ8"}
Keunggulan utama LMDB adalah kemampuannya untuk langsung memetakan file
ke dalam memori. Ini memungkinkan LMDB untuk memberikan penunjuk
langsung ke alamat memori dari kunci dan nilai, menghindari kebutuhan
untuk menyalin data di dalam memori seperti yang biasa dilakukan oleh
basis data lainnya.
:::

::: {#a50b900d .cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="a50b900d" outputId="90334eab-9a07-443c-94c6-6549aef2fef5" scrolled="true"}
``` python
pip install lmdb
```

::: {.output .stream .stdout}
    Requirement already satisfied: lmdb in /usr/local/lib/python3.10/dist-packages (1.4.1)
:::
:::

::: {#6ab11ec0 .cell .markdown id="6ab11ec0"}
## Start With HDF5 ( Hierarchical Data Format) {#start-with-hdf5--hierarchical-data-format}

File HDF terdiri dari dua jenis objek:

Dataset Kelompok
:::

::: {#UQAw3AlxfzZ2 .cell .markdown id="UQAw3AlxfzZ2"}
File HDF5 terdiri dari serangkaian data yang termasuk array
multidimensi, yang dapat menyimpan berbagai ukuran dan jenis data. HDF5
juga memanfaatkan konsep grup, yang merupakan kumpulan dari data-data
ini, memungkinkan organisasi yang lebih terstruktur dan mudah
dimengerti.
:::

::: {#0bcbf4ce .cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="0bcbf4ce" outputId="14d6676c-962c-4570-8d34-2c1b7a947327"}
``` python
pip install h5py
```

::: {.output .stream .stdout}
    Requirement already satisfied: h5py in /usr/local/lib/python3.10/dist-packages (3.9.0)
    Requirement already satisfied: numpy>=1.17.3 in /usr/local/lib/python3.10/dist-packages (from h5py) (1.25.2)
:::
:::

::: {#321c45c1 .cell .markdown id="321c45c1"}
## Menyimpan Satu Gambar
:::

::: {#h59k17qIgKYd .cell .markdown id="h59k17qIgKYd"}
Dengan lima kumpulan CIFAR-10 yang berisi total 50.000 gambar, setiap
gambar dapat digunakan dua kali untuk mencapai total 100.000 gambar.
Diperlukan pembuatan folder untuk menyimpan file gambar dan penentuan
jalur direktori dari tiga variabel. Berikut adalah contoh sintaks untuk
menyimpan gambar:
:::

::: {#28edf59f .cell .code id="28edf59f"}
``` python
from pathlib import Path

disk_dir = Path("data/disk/")
lmdb_dir = Path("data/lmdb/")
hdf5_dir = Path("data/hdf5/")
```
:::

::: {#8S4VsLcdVauo .cell .markdown id="8S4VsLcdVauo"}
**Analisis :** Code di atas menggunakan modul Path dari pustaka pathlib
untuk mendefinisikan tiga jalur direktori yang berbeda: disk_dir,
lmdb_dir, dan hdf5_dir. Ini memudahkan dalam manajemen data dengan
memberikan representasi yang jelas tentang di mana data disimpan atau
diakses dalam sistem file, seperti data yang disimpan di dalam direktori
\"data/disk/\".
:::

::: {#fcff5648 .cell .code id="fcff5648"}
``` python
# membuat folder
disk_dir.mkdir(parents=True, exist_ok=True)
lmdb_dir.mkdir(parents=True, exist_ok=True)
hdf5_dir.mkdir(parents=True, exist_ok=True)
```
:::

::: {#5Xiv_5DPVsTx .cell .markdown id="5Xiv_5DPVsTx"}
**Analisis :** Code di atas menggunakan metode `mkdir()` pada objek
`Path` dari pustaka `pathlib` untuk menciptakan tiga direktori yang
sebelumnya telah ditentukan: `disk_dir`, `lmdb_dir`, dan `hdf5_dir`.
Dengan menggunakan `parents=True`, kode tersebut memastikan bahwa jika
direktori induk belum ada, maka direktori induk akan otomatis dibuat.
Pengaturan `exist_ok=True` memungkinkan penciptaan ulang dari direktori
yang sama tanpa menyebabkan kesalahan. Ini membantu mempermudah
pengelolaan struktur penyimpanan data dan memastikan ketersediaan
direktori yang diperlukan.
:::

::: {#af26cf50 .cell .markdown id="af26cf50"}
## Menyimpan ke Disk
:::

::: {#xRdsLholgOjY .cell .markdown id="xRdsLholgOjY"}
Gambar yang sebelumya terdapat dalam memory dalam bentuk numpy array
akan disimpan ke disk sebagai format png.
:::

::: {#151cfa33 .cell .code id="151cfa33"}
``` python
from PIL import Image
import csv

def store_single_disk(image, image_id, label):
    """ Stores a single image as a .png file on disk.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    Image.fromarray(image).save(disk_dir / f"{image_id}.png")

    with open(disk_dir / f"{image_id}.csv", "wt") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow([label])
```
:::

::: {#0bf25b7f .cell .markdown id="0bf25b7f"}
**Analisis :** Fungsi `store_single_disk()` telah didefinisikan dalam
kode tersebut untuk tujuan menyimpan gambar dalam format `.png` dan
labelnya dalam format `.csv` di dalam direktori yang telah ditentukan
sebelumnya. Modul PIL digunakan untuk mengelola gambar dan modul csv
untuk menulis label. Fungsi ini menerima tiga parameter: `image`, yang
merupakan array gambar dengan ukuran (32, 32, 3), `image_id`, yang
merupakan ID unik untuk gambar, dan `label`, yang merupakan label dari
gambar tersebut. Gambar disimpan dengan nama file yang sesuai dengan ID
gambar dalam format `.png`, sementara label disimpan dalam file `.csv`
yang memiliki nama file yang sama dengan ID gambar. Ini merupakan
pendekatan yang efisien untuk menyimpan data gambar dan labelnya secara
terstruktur di dalam sistem file.
:::

::: {#fcb07e11 .cell .markdown id="fcb07e11"}
## Menyimpan Ke LMDB

LMDB adalah sistem penyimpanan di mana setiap entri disimpan sebagai
array byte. Dalam skenario ini, setiap gambar akan memiliki identifikasi
unik sebagai kunci, dengan gambar itu sendiri sebagai nilai. Kedua nilai
tersebut harus berupa string. Untuk melakukan serialisasi, Anda bisa
menggunakan library \"pickle\", yang memungkinkan semua objek Python
untuk diserialisasi. Oleh karena itu, direkomendasikan untuk menyertakan
metadata gambar dalam basis data.

Metode ini membantu menghindari kesulitan saat perlu menyematkan kembali
metadata ke gambar saat memuat kumpulan data dari disk. Di bawah ini
adalah contoh kode untuk membuat kelas yang menyimpan gambar dan
metadata-nya.
:::

::: {#628d45de .cell .code id="628d45de"}
``` python
class CIFAR_Image:
    def __init__(self, image, label):
        # Dimensions of image for reconstruction - not really necessary
        # for this dataset, but some datasets may include images of
        # varying sizes
        self.channels = image.shape[2]
        self.size = image.shape[:2]

        self.image = image.tobytes()
        self.label = label

    def get_image(self):
        """ Returns the image as a numpy array. """
        image = np.frombuffer(self.image, dtype=np.uint8)
        return image.reshape(*self.size, self.channels)
```
:::

::: {#3d779135 .cell .markdown id="3d779135"}
**Analisis :** Pada code CIFAR_Image dibuat untuk memodelkan
gambar-gambar CIFAR. Pada saat objeknya dibuat, konstruktor
menginisialisasi objek dengan dimensi gambar dan labelnya. Metode
get_image() digunakan untuk mendapatkan gambar dalam bentuk array numpy.
Dalam konstruktor, data gambar diubah menjadi urutan byte untuk
penyimpanan yang efisien, sementara metode get_image() mengembalikannya
ke bentuk aslinya menggunakan numpy.
:::

::: {#isePC4fAiIH2 .cell .markdown id="isePC4fAiIH2"}
Berikut kode untuk menyimpan satu gambar dengan LMDB
:::

::: {#909fdc92 .cell .code id="909fdc92"}
``` python
import lmdb
import pickle

def store_single_lmdb(image, image_id, label):
    """ Stores a single image to a LMDB.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    map_size = image.nbytes * 10

    # Create a new LMDB environment
    env = lmdb.open(str(lmdb_dir / f"single_lmdb"), map_size=map_size)

    # Start a new write transaction
    with env.begin(write=True) as txn:
        # All key-value pairs need to be strings
        value = CIFAR_Image(image, label)
        key = f"{image_id:08}"
        txn.put(key.encode("ascii"), pickle.dumps(value))
    env.close()
```
:::

::: {#7TivVMfGWbM- .cell .markdown id="7TivVMfGWbM-"}
**Analisis :** Fungsi store_single_lmdb() menyimpan gambar dalam file
HDF5. Sebuah file HDF5 baru dibuat dengan nama yang disesuaikan dengan
ID gambar. Kemudian, dataset dibuat di dalam file tersebut untuk
menyimpan gambar dan metadata labelnya. Data gambar dan label disimpan
dalam dataset yang relevan. Setelah proses selesai, file HDF5 ditutup.
:::

::: {#139c3d7e .cell .markdown id="139c3d7e"}
## Storing With HDF5
:::

::: {#ZaE9Q-5-iMui .cell .markdown id="ZaE9Q-5-iMui"}
Berikut kode penyimpanan HDF5 dengan dua data.
:::

::: {#fdd31d94 .cell .code id="fdd31d94"}
``` python
import h5py

def store_single_hdf5(image, image_id, label):
    """ Stores a single image to an HDF5 file.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    # Create a new HDF5 file
    file = h5py.File(hdf5_dir / f"{image_id}.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "image", np.shape(image), h5py.h5t.STD_U8BE, data=image
    )
    meta_set = file.create_dataset(
        "meta", np.shape(label), h5py.h5t.STD_U8BE, data=label
    )
    file.close()
```
:::

::: {#vHiCA2LHW5lv .cell .markdown id="vHiCA2LHW5lv"}
**Analisis :** Fungsi `store_single_hdf5()` berfungsi untuk menyimpan
sebuah gambar ke dalam file HDF5. Dalam fungsi ini, file HDF5 baru
dibuat dengan nama yang sesuai dengan ID gambar. Selanjutnya, dataset
dibuat di dalam file tersebut untuk menyimpan gambar dan metadata
labelnya. Gambar disimpan dalam dataset \"image\", sedangkan label
disimpan dalam dataset \"meta\". Setelah proses selesai, file HDF5
ditutup. Ini adalah pendekatan yang efisien untuk menyimpan data gambar
dan labelnya dalam format HDF5, yang memiliki dukungan untuk struktur
data yang kompleks dan kompresi data.
:::

::: {#71a422ea .cell .markdown id="71a422ea"}
## Storing a single image
:::

::: {#nSulRpQ-iVAg .cell .markdown id="nSulRpQ-iVAg"}
Membuat sebuah dictionary yang mencakup 3 teknik penyimpanan gambar
:::

::: {#1b3298e1 .cell .code id="1b3298e1"}
``` python
_store_single_funcs = dict(
    disk=store_single_disk, lmdb=store_single_lmdb, hdf5=store_single_hdf5
)
```
:::

::: {#1bb7730c .cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="1bb7730c" outputId="3d0dcf1c-40b8-4e0d-eac9-bf7b72dfadfd"}
``` python
from timeit import timeit

store_single_timings = dict()

for method in ("disk", "lmdb", "hdf5"):
    t = timeit(
        "_store_single_funcs[method](image, 0, label)",
        setup="image=images[0]; label=labels[0]",
        number=1,
        globals=globals(),
    )
    store_single_timings[method] = t
    print(f"Method: {method}, Time usage: {t}")
```

::: {.output .stream .stdout}
    Method: disk, Time usage: 0.060984258000189584
    Method: lmdb, Time usage: 0.017186660000334086
    Method: hdf5, Time usage: 0.011729855999874417
:::
:::

::: {#tSzz_GXyicCY .cell .markdown id="tSzz_GXyicCY"}
**Analisis :** Kode tersebut menggunakan modul timeit untuk mengukur
waktu eksekusi dari operasi penyimpanan gambar tunggal menggunakan
metode \"disk\", \"lmdb\", dan \"hdf5\". Melalui iterasi, waktu eksekusi
dari masing-masing metode diukur dan disimpan dalam kamus
`store_single_timings`. Setiap metode dipanggil sekali dengan memilih
fungsi penyimpanan yang sesuai dari kamus `_store_single_funcs`.
Pengukuran waktu ini membantu dalam membandingkan kinerja relatif dari
berbagai metode penyimpanan data gambar. Hasil pengukuran menunjukkan
bahwa metode \"disk\" membutuhkan waktu 0.060984258 detik, \"lmdb\"
membutuhkan waktu 0.017186660 detik, dan \"hdf5\" membutuhkan waktu
0.011729855 detik.
:::

::: {#X1uePZvGmAF1 .cell .markdown id="X1uePZvGmAF1"}
## Storing Many Images
:::

::: {#dndp9M3MixYw .cell .markdown id="dndp9M3MixYw"}
Untuk melakukan menyimpan banyak gambar dilakukan pengkodean sebagai
berikut.
:::

::: {#xjmDnAWXmDEq .cell .code id="xjmDnAWXmDEq"}
``` python
def store_many_disk(images, labels):
    """ Stores an array of images to disk
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)

    # Save all the images one by one
    for i, image in enumerate(images):
        Image.fromarray(image).save(disk_dir / f"{i}.png")

    # Save all the labels to the csv file
    with open(disk_dir / f"{num_images}.csv", "w") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for label in labels:
            # This typically would be more than just one value per row
            writer.writerow([label])

def store_many_lmdb(images, labels):
    """ Stores an array of images to LMDB.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)

    map_size = num_images * images[0].nbytes * 10

    # Create a new LMDB DB for all the images
    env = lmdb.open(str(lmdb_dir / f"{num_images}_lmdb"), map_size=map_size)

    # Same as before — but let's write all the images in a single transaction
    with env.begin(write=True) as txn:
        for i in range(num_images):
            # All key-value pairs need to be Strings
            value = CIFAR_Image(images[i], labels[i])
            key = f"{i:08}"
            txn.put(key.encode("ascii"), pickle.dumps(value))
    env.close()

def store_many_hdf5(images, labels):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)

    # Create a new HDF5 file
    file = h5py.File(hdf5_dir / f"{num_images}_many.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
    )
    meta_set = file.create_dataset(
        "meta", np.shape(labels), h5py.h5t.STD_U8BE, data=labels
    )
    file.close()
```
:::

::: {#xn5hU6vbmF2F .cell .markdown id="xn5hU6vbmF2F"}
## Preparing Dataset
:::

::: {#Igtb_oMfi3Fn .cell .markdown id="Igtb_oMfi3Fn"}
melakukan cutoff yang bernilai 10, 100, 1000, 10000, dan 100000 gambar
:::

::: {#9xw_uxfYmE0o .cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="9xw_uxfYmE0o" outputId="e186b99f-8b22-49a3-e8ad-e087a29994cc"}
``` python
cutoffs = [10, 100, 1000, 10000, 100000]

# Let's double our images so that we have 100,000
images = np.concatenate((images, images), axis=0)
labels = np.concatenate((labels, labels), axis=0)

# Make sure you actually have 100,000 images and labels
print(np.shape(images))
print(np.shape(labels))
```

::: {.output .stream .stdout}
    (100000, 32, 32, 3)
    (100000,)
:::
:::

::: {#47LQ3aX8XaMb .cell .markdown id="47LQ3aX8XaMb"}
**Analisis :** Kode di atas menggandakan jumlah gambar menjadi 100.000
dengan menggabungkan array gambar (images) dan label (labels)
menggunakan metode concatenate() dari NumPy. Dengan mengecek dimensi
array, pastikan jumlah gambar dan label benar-benar menjadi 100.000.
:::

::: {#6hLImLiomLIJ .cell .markdown id="6hLImLiomLIJ"}
## Experiment for Storing Many Images
:::

::: {#uNyu5Jd9i792 .cell .markdown id="uNyu5Jd9i792"}
setelah itu melakukan eksperiment untuk menyimpan banyak gambar
:::

::: {#xzm_ta4_mO6r .cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="xzm_ta4_mO6r" outputId="823d6c40-5f2f-4c97-d54a-393104b8c1d3"}
``` python
_store_many_funcs = dict(
    disk=store_many_disk, lmdb=store_many_lmdb, hdf5=store_many_hdf5
)

from timeit import timeit

store_many_timings = {"disk": [], "lmdb": [], "hdf5": []}

for cutoff in cutoffs:
    for method in ("disk", "lmdb", "hdf5"):
        t = timeit(
            "_store_many_funcs[method](images_, labels_)",
            setup="images_=images[:cutoff]; labels_=labels[:cutoff]",
            number=1,
            globals=globals(),
        )
        store_many_timings[method].append(t)

        # Print out the method, cutoff, and elapsed time
        print(f"Method: {method}, Time usage: {t}")
```

::: {.output .stream .stdout}
    Method: disk, Time usage: 0.012574804999985645
    Method: lmdb, Time usage: 0.0074170439997942594
    Method: hdf5, Time usage: 0.00227957700008119
    Method: disk, Time usage: 0.06729665400007434
    Method: lmdb, Time usage: 0.013505107000128191
    Method: hdf5, Time usage: 0.0030577219999941008
    Method: disk, Time usage: 0.6516917849999118
    Method: lmdb, Time usage: 0.06246709100014414
    Method: hdf5, Time usage: 0.008764534999954776
    Method: disk, Time usage: 5.286024412000188
    Method: lmdb, Time usage: 0.464327479000076
    Method: hdf5, Time usage: 0.07172583699957613
    Method: disk, Time usage: 55.601886037999975
    Method: lmdb, Time usage: 8.308399344000009
    Method: hdf5, Time usage: 0.8323716119998608
:::
:::

::: {#0TgS6IpUjDWF .cell .markdown id="0TgS6IpUjDWF"}
penyimpanan gambar disimpan 5 kali. untuk memilih model yang terbaik
dapat di lakukan pengecakan dari ketiga metode tersebut pada setiap
cutoff, metode yang terbaik ialah metode dengan time usage yang lebih
kecil.
:::

::: {#499TCxibmQ8Y .cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":1000}" id="499TCxibmQ8Y" outputId="f5e98912-f8e8-4618-b8b7-d824abb837e3"}
``` python
import matplotlib.pyplot as plt

def plot_with_legend(
    x_range, y_data, legend_labels, x_label, y_label, title, log=False
):
    """ Displays a single plot with multiple datasets and matching legends.
        Parameters:
        --------------
        x_range         list of lists containing x data
        y_data          list of lists containing y values
        legend_labels   list of string legend labels
        x_label         x axis label
        y_label         y axis label
    """
    plt.style.use("seaborn-whitegrid")
    plt.figure(figsize=(10, 7))

    if len(y_data) != len(legend_labels):
        raise TypeError(
            "Error: number of data sets does not match number of labels."
        )

    all_plots = []
    for data, label in zip(y_data, legend_labels):
        if log:
            temp, = plt.loglog(x_range, data, label=label)
        else:
            temp, = plt.plot(x_range, data, label=label)
        all_plots.append(temp)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(handles=all_plots)
    plt.show()

# Getting the store timings data to display
disk_x = store_many_timings["disk"]
lmdb_x = store_many_timings["lmdb"]
hdf5_x = store_many_timings["hdf5"]

plot_with_legend(
    cutoffs,
    [disk_x, lmdb_x, hdf5_x],
    ["PNG files", "LMDB", "HDF5"],
    "Number of images",
    "Seconds to store",
    "Storage time",
    log=False,
)

plot_with_legend(
    cutoffs,
    [disk_x, lmdb_x, hdf5_x],
    ["PNG files", "LMDB", "HDF5"],
    "Number of images",
    "Seconds to store",
    "Log storage time",
    log=True,
)
```

::: {.output .stream .stderr}
    <ipython-input-31-99d89538a067>:15: MatplotlibDeprecationWarning: The seaborn styles shipped by Matplotlib are deprecated since 3.6, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<style>'. Alternatively, directly use the seaborn API instead.
      plt.style.use("seaborn-whitegrid")
:::

::: {.output .display_data}
![](vertopal_ed03a3e8559046e18726246478f5e3e8/53192b40249275040cc76a91dc900913b878325e.png)
:::

::: {.output .stream .stderr}
    <ipython-input-31-99d89538a067>:15: MatplotlibDeprecationWarning: The seaborn styles shipped by Matplotlib are deprecated since 3.6, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<style>'. Alternatively, directly use the seaborn API instead.
      plt.style.use("seaborn-whitegrid")
:::

::: {.output .display_data}
![](vertopal_ed03a3e8559046e18726246478f5e3e8/ddff63d4a825c9f885b9ae38c1cbb7a0fda10dc3.png)
:::
:::

::: {#Jhpq9YlOXrPY .cell .markdown id="Jhpq9YlOXrPY"}
**Analisis :** Kode tersebut menghasilkan fungsi `plot_with_legend()`
yang bertujuan untuk menampilkan plot tunggal dengan beberapa dataset
dan legenda yang sesuai. Dalam kasus ini, fungsi tersebut digunakan
untuk memplot data waktu penyimpanan dari tiga metode yang berbeda:
\"PNG files\", \"LMDB\", dan \"HDF5\", terhadap jumlah gambar yang
bervariasi. Plot pertama menampilkan waktu penyimpanan dalam skala
linear, sedangkan plot kedua menampilkan waktu penyimpanan dalam skala
logaritmik. Hal ini memungkinkan visualisasi perbandingan waktu
penyimpanan antara metode-metode tersebut.
:::

::: {#_mDyG_PHmXHi .cell .markdown id="_mDyG_PHmXHi"}
## Reading a Single Images
:::

::: {#BXpWph0yjex- .cell .markdown id="BXpWph0yjex-"}
Untuk dapat membaca dari disk dengan format png digunakan syntax berikut
:::

::: {#Sjm20-GJmZ7U .cell .code id="Sjm20-GJmZ7U"}
``` python
def read_single_disk(image_id):
    """ Stores a single image to disk.
        Parameters:
        ---------------
        image_id    integer unique ID for image

        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    image = np.array(Image.open(disk_dir / f"{image_id}.png"))

    with open(disk_dir / f"{image_id}.csv", "r") as csvfile:
        reader = csv.reader(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        label = int(next(reader)[0])

    return image, label
```
:::

::: {#Zb-wLgO7Xyg_ .cell .markdown id="Zb-wLgO7Xyg_"}
**Analisis :** Fungsi `read_single_disk()` bertujuan untuk membaca
sebuah gambar beserta labelnya dari disk. Proses membaca gambar
melibatkan penggunaan modul Image dari PIL dengan menggunakan fungsi
`open()` untuk membuka gambar dan mengonversinya menjadi array numpy.
Labelnya dibaca dari file .csv yang sesuai dengan ID gambar. Setelah
proses membaca gambar dan labelnya selesai, fungsi mengembalikan gambar
dan label tersebut.
:::

::: {#0c8xQsBWmcUI .cell .markdown id="0c8xQsBWmcUI"}
## Reading From LMDB
:::

::: {#f6MMRKZNjide .cell .markdown id="f6MMRKZNjide"}
Untuk proses read dengan metode LMBD
:::

::: {#p5BEkf0ameas .cell .code id="p5BEkf0ameas"}
``` python
def read_single_lmdb(image_id):
    """ Stores a single image to LMDB.
        Parameters:
        ---------------
        image_id    integer unique ID for image

        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    # Open the LMDB environment
    env = lmdb.open(str(lmdb_dir / f"single_lmdb"), readonly=True)

    # Start a new read transaction
    with env.begin() as txn:
        # Encode the key the same way as we stored it
        data = txn.get(f"{image_id:08}".encode("ascii"))
        # Remember it's a CIFAR_Image object that is loaded
        cifar_image = pickle.loads(data)
        # Retrieve the relevant bits
        image = cifar_image.get_image()
        label = cifar_image.label
    env.close()

    return image, label
```
:::

::: {#SuS7Cm3qmgkq .cell .markdown id="SuS7Cm3qmgkq"}
## Reading From HDF5
:::

::: {#nRHN8_JDmjDN .cell .code id="nRHN8_JDmjDN"}
``` python
def read_single_hdf5(image_id):
    """ Stores a single image to HDF5.
        Parameters:
        ---------------
        image_id    integer unique ID for image

        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    # Open the HDF5 file
    file = h5py.File(hdf5_dir / f"{image_id}.h5", "r+")

    image = np.array(file["/image"]).astype("uint8")
    label = int(np.array(file["/meta"]).astype("uint8"))

    return image, label
```
:::

::: {#7-lifHTSmlt2 .cell .code id="7-lifHTSmlt2"}
``` python
_read_single_funcs = dict(
    disk=read_single_disk, lmdb=read_single_lmdb, hdf5=read_single_hdf5
)
```
:::

::: {#XeXkLOA1mqk6 .cell .markdown id="XeXkLOA1mqk6"}
## Experiment for Reading a Single Image
:::

::: {#fEvleGoVmvIO .cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="fEvleGoVmvIO" outputId="1af70eb0-6337-47fa-ca44-7af2679c1f9d"}
``` python
from timeit import timeit

read_single_timings = dict()

for method in ("disk", "lmdb", "hdf5"):
    t = timeit(
        "_read_single_funcs[method](0)",
        setup="image=images[0]; label=labels[0]",
        number=1,
        globals=globals(),
    )
    read_single_timings[method] = t
    print(f"Method: {method}, Time usage: {t}")
```

::: {.output .stream .stdout}
    Method: disk, Time usage: 0.002999725000336184
    Method: lmdb, Time usage: 0.007998118000159593
    Method: hdf5, Time usage: 0.002849938000053953
:::
:::

::: {#iYDRvwxDjpTj .cell .markdown id="iYDRvwxDjpTj"}
**Analisis :** Dari code di atas didapatkan hasil bahwa proses read
gambar dengan metode Disk menghabiskan waktu 0.002999725000336184 detik,
LMBD 0.007998118000159593 detik dan HDF5 0.0.002849938000053953 detik.
:::

::: {#7ggN0VwLmyuM .cell .markdown id="7ggN0VwLmyuM"}
## Reading Many Images
:::

::: {#euMZczOqm2nA .cell .markdown id="euMZczOqm2nA"}
### Adjusting the Code for Many Images
:::

::: {#UlFfJUz-m6tw .cell .code id="UlFfJUz-m6tw"}
``` python
def read_many_disk(num_images):
    """ Reads image from disk.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []

    # Loop over all IDs and read each image in one by one
    for image_id in range(num_images):
        images.append(np.array(Image.open(disk_dir / f"{image_id}.png")))

    with open(disk_dir / f"{num_images}.csv", "r") as csvfile:
        reader = csv.reader(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for row in reader:
            labels.append(int(row[0]))
    return images, labels

def read_many_lmdb(num_images):
    """ Reads image from LMDB.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []
    env = lmdb.open(str(lmdb_dir / f"{num_images}_lmdb"), readonly=True)

    # Start a new read transaction
    with env.begin() as txn:
        # Read all images in one single transaction, with one lock
        # We could split this up into multiple transactions if needed
        for image_id in range(num_images):
            data = txn.get(f"{image_id:08}".encode("ascii"))
            # Remember that it's a CIFAR_Image object
            # that is stored as the value
            cifar_image = pickle.loads(data)
            # Retrieve the relevant bits
            images.append(cifar_image.get_image())
            labels.append(cifar_image.label)
    env.close()
    return images, labels

def read_many_hdf5(num_images):
    """ Reads image from HDF5.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []

    # Open the HDF5 file
    file = h5py.File(hdf5_dir / f"{num_images}_many.h5", "r+")

    images = np.array(file["/images"]).astype("uint8")
    labels = np.array(file["/meta"]).astype("uint8")

    return images, labels

_read_many_funcs = dict(
    disk=read_many_disk, lmdb=read_many_lmdb, hdf5=read_many_hdf5
)
```
:::

::: {#54h0BHN9m89H .cell .markdown id="54h0BHN9m89H"}
### Experiment for Reading Many Images
:::

::: {#c1SfJhv_nAeW .cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="c1SfJhv_nAeW" outputId="8f0ffaaf-5490-4b52-d445-0d7ef0f17a56"}
``` python
from timeit import timeit

read_many_timings = {"disk": [], "lmdb": [], "hdf5": []}

for cutoff in cutoffs:
    for method in ("disk", "lmdb", "hdf5"):
        t = timeit(
            "_read_many_funcs[method](num_images)",
            setup="num_images=cutoff",
            number=1,
            globals=globals(),
        )
        read_many_timings[method].append(t)

        # Print out the method, cutoff, and elapsed time
        print(f"Method: {method}, No. images: {cutoff}, Time usage: {t}")
```

::: {.output .stream .stdout}
    Method: disk, No. images: 10, Time usage: 0.011922941000193532
    Method: lmdb, No. images: 10, Time usage: 0.0012311049999880197
    Method: hdf5, No. images: 10, Time usage: 0.004308608999963326
    Method: disk, No. images: 100, Time usage: 0.09547990300006859
    Method: lmdb, No. images: 100, Time usage: 0.0027350039999873843
    Method: hdf5, No. images: 100, Time usage: 0.0032038920003287785
    Method: disk, No. images: 1000, Time usage: 0.6230934780001007
    Method: lmdb, No. images: 1000, Time usage: 0.023417476999838982
    Method: hdf5, No. images: 1000, Time usage: 0.008918160000121134
    Method: disk, No. images: 10000, Time usage: 5.777562597999804
    Method: lmdb, No. images: 10000, Time usage: 0.1727316989999963
    Method: hdf5, No. images: 10000, Time usage: 0.03943293600013931
    Method: disk, No. images: 100000, Time usage: 31.413787462000073
    Method: lmdb, No. images: 100000, Time usage: 1.4483320590002222
    Method: hdf5, No. images: 100000, Time usage: 1.3208534979999058
:::
:::

::: {#JYski0q6j_Mi .cell .markdown id="JYski0q6j_Mi"}
dapat dilihat bahwa metode lmdb lebih cepat membaca image ketimbang
metode disk dan hdfs
:::

::: {#jq12Uil3nDD2 .cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":1000}" id="jq12Uil3nDD2" outputId="5d01c667-5c85-4d32-d69b-81dc79ec5ff4"}
``` python
disk_x_r = read_many_timings["disk"]
lmdb_x_r = read_many_timings["lmdb"]
hdf5_x_r = read_many_timings["hdf5"]

plot_with_legend(
    cutoffs,
    [disk_x_r, lmdb_x_r, hdf5_x_r],
    ["PNG files", "LMDB", "HDF5"],
    "Number of images",
    "Seconds to read",
    "Read time",
    log=False,
)

plot_with_legend(
    cutoffs,
    [disk_x_r, lmdb_x_r, hdf5_x_r],
    ["PNG files", "LMDB", "HDF5"],
    "Number of images",
    "Seconds to read",
    "Log read time",
    log=True,
)
```

::: {.output .stream .stderr}
    <ipython-input-32-99d89538a067>:15: MatplotlibDeprecationWarning: The seaborn styles shipped by Matplotlib are deprecated since 3.6, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<style>'. Alternatively, directly use the seaborn API instead.
      plt.style.use("seaborn-whitegrid")
:::

::: {.output .display_data}
![](vertopal_ed03a3e8559046e18726246478f5e3e8/c7c94357ea33bb68a9a3b3adfb9ef36056f13b0e.png)
:::

::: {.output .stream .stderr}
    <ipython-input-32-99d89538a067>:15: MatplotlibDeprecationWarning: The seaborn styles shipped by Matplotlib are deprecated since 3.6, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<style>'. Alternatively, directly use the seaborn API instead.
      plt.style.use("seaborn-whitegrid")
:::

::: {.output .display_data}
![](vertopal_ed03a3e8559046e18726246478f5e3e8/f78963cb98eb5417307d4ca1dae7fee899dd5d74.png)
:::
:::

::: {#5xQ5yEaRnGZI .cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":691}" id="5xQ5yEaRnGZI" outputId="9c182587-5cbf-46fe-f642-ef455513d6f4"}
``` python
plot_with_legend(
    cutoffs,
    [disk_x_r, lmdb_x_r, hdf5_x_r, disk_x, lmdb_x, hdf5_x],
    [
        "Read PNG",
        "Read LMDB",
        "Read HDF5",
        "Write PNG",
        "Write LMDB",
        "Write HDF5",
    ],
    "Number of images",
    "Seconds",
    "Log Store and Read Times",
    log=False,
)
```

::: {.output .stream .stderr}
    <ipython-input-32-99d89538a067>:15: MatplotlibDeprecationWarning: The seaborn styles shipped by Matplotlib are deprecated since 3.6, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<style>'. Alternatively, directly use the seaborn API instead.
      plt.style.use("seaborn-whitegrid")
:::

::: {.output .display_data}
![](vertopal_ed03a3e8559046e18726246478f5e3e8/ea371fea5b11ca5127a29e2542efd0b507fa5615.png)
:::
:::

::: {#IqZnra-ykcvz .cell .markdown id="IqZnra-ykcvz"}
**Kesimpulan :** Sistem penyimpanan seperti format PNG, LMDB, dan HDF5
memiliki kombinasi kelemahan dan keunggulan yang berbeda. Misalnya, LMDB
menonjol dengan kinerja tinggi dan integritas data yang kuat karena
kemampuannya untuk menulis data baru tanpa mengganggu atau memindahkan
data yang ada. Pendekatan ini menghasilkan operasi pembacaan yang cepat
dan dapat diandalkan tanpa memerlukan penyimpanan log transaksi
tambahan. Di sisi lain, HDF5 memberikan keunggulan dalam situasi yang
membutuhkan kinerja tinggi, terutama ketika mengakses rentang besar
dalam kumpulan data. Struktur penyimpanan HDF5 memungkinkan akses cepat
ke rentang data, mengurangi overhead yang terkait dengan membaca setiap
elemen secara terpisah. Pemahaman yang mendalam tentang struktur dan
karakteristik dari masing-masing format penyimpanan sangat penting untuk
mengoptimalkan kinerja aplikasi sehari-hari dan memilih metode yang
paling sesuai dengan kebutuhan spesifik.
:::

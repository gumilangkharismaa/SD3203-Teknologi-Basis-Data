# **Three Ways of Storing and Accessing Lots of Images in Python**

# Setup
## A Dataset to Play With
 Kumpulan data gambar Canadian Institute for Advance Research, dikenal sebagai CIFAR-10, Terdapat 60.000 gambar berwarna berukuran 32x32 piksel yang termasuk dalam objek yang berbeda, seperti anjing, kucing, dan pesawat terbang. Saat mengunduh dan mengekstrak folder tersebut, akan ditemukan bahwa file tersebut bukanlah file gambar yang dapat dibaca manusia. Mereka sebenarnya telah diserialkan dan disimpan dalam batch menggunakan cPickle. selain mengekstrak dataset CIFAR, perlu disebutkan bahwa modul acar Python memiliki keuntungan utama karena dapat membuat serialisasi objek Python apa pun tanpa kode tambahan atau transformasi apa pun.

## Setup for Storing Images on Disk
1. **Instal Python 3.x**: Menginstall python jikau memang belum ada.

2. **Instal Pillow**: Gunakan Pillow untuk manipulasi gambar. Anda dapat menginstalnya menggunakan pip install Pillow.

## Getting Started With LMDB
LMDB, terkadang disebut sebagai “Lightning Database,” adalah singkatan dari Lightning Memory-Mapped Database karena cepat dan menggunakan file yang dipetakan memori. Ini adalah penyimpanan nilai kunci, bukan database relasional. Dalam hal implementasi, LMDB adalah pohon B+, yang pada dasarnya berarti struktur grafik mirip pohon yang disimpan dalam memori di mana setiap elemen nilai kunci adalah sebuah simpul, dan simpul dapat memiliki banyak anak. Node pada level yang sama dihubungkan satu sama lain untuk traversal yang cepat. Alasan utama lainnya mengapa LMDB efisien adalah karena LMDB dipetakan dalam memori. Artinya, ia mengembalikan penunjuk langsung ke alamat memori dari kunci dan nilai, tanpa perlu menyalin apa pun di memori seperti yang dilakukan kebanyakan database lainnya.

## Getting Started With HDF5
HDF5 adalah singkatan dari Hierarchical Data Format, format file yang disebut sebagai HDF4 atau HDF5. Kita tidak perlu khawatir tentang HDF4, karena HDF5 adalah versi yang dipertahankan saat ini. Menariknya, HDF berasal dari National Center for Supercomputing Applications, sebagai format data ilmiah yang portabel dan ringkas. Jika Anda bertanya-tanya apakah ini digunakan secara luas, lihat uraian singkat NASA tentang HDF5 dari proyek Data Bumi mereka. File HDF terdiri dari dua jenis objek , yaitu Kumpulan data
dan Grup. Kumpulan data adalah array multidimensi, dan grup terdiri dari kumpulan data atau grup lain. Array multidimensi dengan ukuran dan tipe apa pun dapat disimpan sebagai kumpulan data, namun dimensi dan tipenya harus seragam dalam kumpulan data. Setiap dataset harus berisi array berdimensi N yang homogen.


# Storing a Single Image

1. **Membuat Direktori untuk Setiap Metode Penyimpanan**:  
   Buatlah tiga direktori terpisah untuk setiap metode penyimpanan yang akan diuji, yaitu sistem file standar, LMDB, dan format HDF5.  
   ```bash
   mkdir data/disk
   mkdir data/lmdb
   mkdir data/hdf5
   ```
2. **Menyimpan Jalur Direktori ke dalam Variabel Python**:
```python
from pathlib import Path
disk_dir = Path("data/disk/")
lmdb_dir = Path("data/lmdb/")
hdf5_dir = Path("data/hdf5/")
```
3. **Persiapan Data Gambar untuk Eksperimen**
Jika menggunakan dataset CIFAR-10 yang terdiri dari 50,000 gambar, gunakan setiap gambar dua kali untuk mendapatkan total 100,000 gambar dalam eksperimen.
4. **Menyusun Eksperimen dengan Berbagai Jumlah File**
Bandingkan kinerja metode penyimpanan dengan menguji berbagai jumlah gambar, mulai dari satu gambar hingga 100,000 gambar.

## Storing to Disk
menyimpannya terlebih dahulu ke disk sebagai gambar .png, dan beri nama menggunakan ID gambar unik image_id. Ini dapat dilakukan dengan menggunakan paket Pillow yang Anda instal sebelumnya:

```python
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
Ini akan menyimpan gambar. Dalam semua aplikasi realistis, Anda juga peduli dengan meta data yang dilampirkan pada gambar, yang dalam contoh kumpulan data kami adalah label gambar. Saat Anda menyimpan gambar ke disk, ada beberapa opsi untuk menyimpan data meta.

## Storing to LMDB
Pertama, LMDB adalah sistem penyimpanan nilai kunci di mana setiap entri disimpan sebagai array byte, jadi dalam kasus kita, kunci akan menjadi pengidentifikasi unik untuk setiap gambar, dan nilainya akan menjadi gambar itu sendiri. Baik kunci maupun nilai diharapkan berupa string, jadi penggunaan umum adalah membuat serial nilai sebagai string, lalu membatalkan serialisasinya saat membacanya kembali.

Anda dapat menggunakan acar untuk membuat serialisasi. Objek Python apa pun dapat dibuat serial, jadi sebaiknya Anda juga menyertakan data meta gambar ke dalam database. Ini menyelamatkan Anda dari kesulitan melampirkan meta data kembali ke data gambar saat kami memuat kumpulan data dari disk.

**Membuat Kelas Python untuk Gambar dan Meta Data**
```python
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
Kedua, karena LMDB dipetakan dengan memori, database baru perlu mengetahui berapa banyak memori yang diperkirakan akan digunakan. Hal ini relatif mudah dalam kasus kami, namun dapat menjadi masalah besar dalam kasus lain, yang akan Anda lihat lebih mendalam di bagian selanjutnya. LMDB menyebut variabel ini sebagai map_size.

Terakhir, operasi baca dan tulis dengan LMDB dilakukan dalam transaksi. Anda dapat menganggapnya mirip dengan database tradisional, yang terdiri dari sekelompok operasi pada database. Ini mungkin terlihat jauh lebih rumit daripada versi disk.

**Membuat kode untuk save satu foto untuk LMDB**
```python
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


## Storing With HDF5
Ingatlah bahwa file HDF5 dapat berisi lebih dari satu dataset. Ini memungkinkan penyimpanan yang efisien dari berbagai jenis data. Anda akan membuat dua dataset dalam file HDF5: satu untuk gambar dan satu lagi untuk meta data gambar. Gunakan `h5py.h5t.STD_U8BE` untuk menentukan tipe data yang akan disimpan dalam dataset. Tipe data ini adalah integer 8-bit tak bertanda. Tipe data ini dipilih karena sesuai untuk menyimpan gambar yang diwakili dalam bentuk pixel values yang berkisar dari 0 sampai 255.
Pilihan tipe data sangat mempengaruhi kebutuhan runtime dan penyimpanan dari file HDF5.Pilihlah tipe data yang memenuhi kebutuhan minimal Anda untuk mengoptimalkan penggunaan sumber daya.

```python
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

## Experiments for Storing a Single Image
Simpan semua fungsi penyimpanan dalam satu kamus untuk memudahkan pemanggilan selama eksperimen waktu.
```python
_store_single_funcs = dict(
    disk=store_single_disk, lmdb=store_single_lmdb, hdf5=store_single_hdf5
)
```

Jalankan eksperimen untuk menyimpan gambar pertama dari CIFAR dan labelnya dalam tiga cara yang berbeda: Disk, LMDB, dan HDF5. Catat waktu runtime dan penggunaan memori untuk setiap metode.

```python
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

# Storing Many Images
## Adjusting the Code for Many Images
**Metode Penyimpanan File .png Sederhana**:
Untuk menyimpan banyak gambar sebagai file .png, prosesnya cukup sederhana, yaitu dengan memanggil metode `store_single_method()` berulang kali untuk setiap gambar Pendekatan ini memungkinkan setiap gambar disimpan dalam file terpisah dengan format .png.

**Pendekatan Penyimpanan LMDB dan HDF5**:
Untuk format penyimpanan seperti LMDB (Lightning Memory-Mapped Database) atau HDF5 (Hierarchical Data Format), pendekatan penyimpanan berbeda. Alih-alih membuat file database atau file HDF5 terpisah untuk setiap gambar, lebih efisien untuk menyimpan semua gambar dalam satu atau beberapa file saja. Pendekatan ini memungkinkan pengelolaan data yang lebih efisien dan menghindari fragmentasi berkas yang berlebihan. Anda perlu mengubah kode dan membuat tiga fungsi baru yang menerima beberapa gambar: `store_many_disk()`, `store_many_lmdb()`, dan `store_many_hdf5()`. `store_many_disk()`, Metode ini diubah untuk melakukan loop atas setiap gambar dalam daftar dan menyimpannya sebagai file terpisah `store_many_lmdb()`, Loop juga diperlukan di sini karena kita membuat objek `CIFAR_Image` untuk setiap gambar dan meta datanya.
`store_many_hdf5()`, Penyesuaian terkecil ada pada metode ini. Faktanya, hampir tidak ada penyesuaian sama sekali! File HDF5 tidak memiliki batasan ukuran file selain pembatasan eksternal atau ukuran dataset, sehingga semua gambar dimasukkan ke dalam satu dataset, sama seperti sebelumnya.

```python
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
## Preparing the Dataset
Sebelum menjalankan eksperimen lagi, mari kita pertama kali menggandakan ukuran dataset kita agar kita dapat menguji dengan hingga 100.000 gambar. 
```python
cutoffs = [10, 100, 1000, 10000, 100000]

# Let's double our images so that we have 100,000
images = np.concatenate((images, images), axis=0)
labels = np.concatenate((labels, labels), axis=0)

# Make sure you actually have 100,000 images and labels
print(np.shape(images))
print(np.shape(labels))
```
Setelah menggandakan ukuran dataset, Anda dapat menjalankan eksperimen dengan dataset yang baru saja diperbesar untuk menguji performa metode penyimpanan dengan jumlah gambar hingga 100.000.

## Experiment for Storing Many Images
```python
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
Perlu bersabar sejenak dan menunggu dengan penuh penasaran sementara 111,110 gambar disimpan tiga kali masing-masing ke disk Anda, dalam tiga format yang berbeda. Anda juga perlu bersiap untuk mengucapkan selamat tinggal pada sekitar 2 GB ruang disk.


# Reading a Single Image
LMDB memerlukan kerja keras paling besar saat membaca kembali file gambar dari memori, karena langkah serialisasi. Buka file gambar (.png) menggunakan identifikasi gambar (`image_id`). Buka dan baca file CSV untuk menemukan metadata yang sesuai dengan `image_id`.

```python
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
## Reading From LMDB
Buka database LMDB menggunakan path yang diberikan. Gunakan `image_id` untuk mendapatkan data gambar yang diserialkan. Deserialkan data untuk mendapatkan gambar dan metadata.

```python
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

## Reading From HDF5
Membaca dengan membuka file HDF5. Akses dataset `images` untuk mendapatkan gambar, dan dataset `labels` untuk metadata, menggunakan indeks gambar (`image_index`).

```python
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

_read_single_funcs = dict(
    disk=read_single_disk, lmdb=read_single_lmdb, hdf5=read_single_hdf5
)
```

## Experiment for Reading a Single Image


```python
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

4. Kesimpulan
**Kecepatan Pembacaan File .png dan .csv dari Disk**:
Membaca file .png (gambar) dan .csv (metadata) secara langsung dari disk sedikit lebih cepat dibandingkan dengan menggunakan metode penyimpanan lainnya seperti LMDB atau HDF5.
Kecepatan ini diperoleh karena pembacaan file langsung dari sistem file disk umumnya merupakan operasi yang cukup cepat. Meskipun pembacaan file .png dan .csv dari disk sedikit lebih cepat, ketiga metode penyimpanan (disk, LMDB, dan HDF5) secara keseluruhan memberikan performa yang cepat dan hampir sama. Perbedaan kecepatan antara ketiga metode ini dianggap tidak signifikan atau tidak terlalu besar.


# Reading Many Images
## Adjusting the Code for Many Images
Memperluas fungsi di atas, Anda dapat membuat fungsi dengan read_many_, yang dapat digunakan untuk percobaan berikutnya. Seperti sebelumnya, menarik untuk membandingkan kinerja saat membaca jumlah gambar yang berbeda, yang diulangi dalam kode di bawah ini untuk referensi:

```python
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

# Experiment for Reading Many Images
```python
from timeit import timeit

read_many_timings = {"disk": [], "lmdb": [], "hdf5": []}

for cutoff in cutoffs:
    for method in ("disk", "lmdb", "hdf5"):
        t = timeit(
            "_read_many_funcs[method](num_images)",
            setup="num_images=cutoff",
            number=0,
            globals=globals(),
        )
        read_many_timings[method].append(t)

        # Print out the method, cutoff, and elapsed time
        print(f"Method: {method}, No. images: {cutoff}, Time usage: {t}")

```

# Discussion
Ada fitur pembeda lainnya dari LMDB dan HDF5 yang perlu diketahui, dan penting juga untuk membahas secara singkat beberapa kritik terhadap kedua metode tersebut.

## Parallel Access
Perbandingan utama yang tidak kami uji dalam eksperimen di atas adalah pembacaan dan penulisan secara bersamaan. Seringkali, dengan kumpulan data sebesar itu, Anda mungkin ingin mempercepat operasi Anda melalui paralelisasi. Beberapa aplikasi dapat mengakses database LMDB yang sama secara bersamaan, dan beberapa thread dari proses yang sama juga dapat mengakses LMDB secara bersamaan untuk dibaca. Hal ini memungkinkan waktu baca yang lebih cepat: jika Anda membagi seluruh CIFAR menjadi sepuluh set, maka Anda dapat menyiapkan sepuluh proses untuk setiap pembacaan dalam satu set, dan ini akan membagi waktu pemuatan menjadi sepuluh. HDF5 juga menawarkan I/O paralel, memungkinkan pembacaan dan penulisan secara bersamaan. Namun, dalam implementasinya, kunci tulis ditahan, dan akses dilakukan secara berurutan, kecuali Anda memiliki sistem file paralel.

Ada dua opsi utama jika Anda mengerjakan sistem seperti itu, yang dibahas lebih mendalam dalam artikel ini oleh Grup HDF tentang IO paralel. Ini bisa menjadi sangat rumit, dan opsi paling sederhana adalah membagi kumpulan data Anda menjadi beberapa file HDF5 secara cerdas, sehingga setiap proses dapat menangani satu file .h5 secara terpisah dari yang lain.

## A More Critical Look at Implementation
Tidak ada utopia dalam sistem penyimpanan, dan baik LMDB maupun HDF5 memiliki kelemahan masing-masing. Hal penting yang perlu dipahami tentang LMDB adalah bahwa data baru ditulis tanpa menimpa atau memindahkan data yang sudah ada. Ini adalah keputusan desain yang memungkinkan pembacaan sangat cepat yang Anda saksikan dalam eksperimen kami, dan juga menjamin integritas dan keandalan data tanpa perlu lagi menyimpan log transaksi. menentukan parameter map_size untuk alokasi memori sebelum menulis ke database baru? Di sinilah LMDB bisa merepotkan. Misalkan Anda telah membuat database LMDB, dan semuanya baik-baik saja. Anda telah menunggu dengan sabar hingga kumpulan data Anda yang sangat besar dimasukkan ke dalam LMDB.

## Conclusion
Dalam artikel ini, Anda telah diperkenalkan dengan tiga cara menyimpan dan mengakses banyak gambar dengan Python, dan mungkin berkesempatan untuk mencoba beberapa di antaranya. Semua kode untuk artikel ini ada di notebook Jupyter di sini atau skrip Python di sini. Anda telah melihat bukti bagaimana berbagai metode penyimpanan dapat memengaruhi waktu baca dan tulis secara drastis, serta beberapa pro dan kontra dari ketiga metode yang dibahas dalam artikel ini. Meskipun menyimpan gambar sebagai file .png mungkin merupakan cara yang paling intuitif, ada manfaat kinerja yang besar jika mempertimbangkan metode seperti HDF5 atau LMDB.

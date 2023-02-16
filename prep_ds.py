"""Prepare the dataset."""

import imghdr
import os
import shutil
import tarfile

import gdown
import kaggle
import tensorflow as tf

import oi_download

gic_dir = os.path.join(os.getcwd(), "gic_dataset")


def take_from_ds(unzipped_dir, dataset_slice_dirs: dict, ds_map: dict):
    for s_slice_dir, d_slice_dir in dataset_slice_dirs.items():
        src_dir = os.path.join(unzipped_dir, s_slice_dir)
        dst_dir = os.path.join(gic_dir, d_slice_dir)
        copy_map_dirs(src_dir, dst_dir, ds_map)


def copy_map_dirs(src_dir, dst_dir, ds_map: dict):
    for src, dst in ds_map.items():
        src_path = os.path.join(src_dir, src)
        dst_path = os.path.join(dst_dir, dst)

        if os.path.exists(dst_path):
            for fname in os.listdir(src_path):
                shutil.copy2(os.path.join(src_path, fname),
                             os.path.join(dst_path, src + fname))
        else:
            shutil.copytree(src_path, dst_path)


def take_from_oi(oi_map: dict, oi_output_map: dict):
    for oi_label, gic_label in oi_map.items():
        dst_dir = os.path.join(gic_dir, "train", gic_label)
        shutil.move(oi_output_map[oi_label.lower()], dst_dir)


def prep_ds():
    ims_dir = os.path.join(os.getcwd(), "intermediates")
    os.mkdir(ims_dir)
    os.chdir(ims_dir)

    # camsdd ds
    ds_url = "https://data.vision.ee.ethz.ch//ihnatova/public/camsdd/CamSDD.zip"
    zip_path = tf.keras.utils.get_file(fname="camsdd.zip",
                                       origin=ds_url,
                                       extract=True)
    cam_sdd_map = {
        "1_Portrait": "Selfies",
        "2_Group_portrait": "Group portraits",
        "3_Kids": "Kids",
        "4_Dog": "Dogs",
        "5_Cat": "Cats",
        "6_Macro": "Macro shots",
        "7_Food": "Food",
        "8_Beach": "Beach",
        "9_Mountain": "Hills",
        "10_Waterfall": "Waterfalls",
        "11_Snow": "Snow",
        "12_Landscape": "Landscapes",
        "13_Underwater": "Waters",
        "14_Architecture": "Architecture",
        "15_Sunset_Sunrise": "Landscapes",
        "16_Blue_Sky": "Blue Sky",
        "17_Cloudy_Sky": "Cloudy Sky",
        "18_Greenery": "Greenery",
        "19_Autumn_leaves": "Woods",
        "20_Flower": "Flowers",
        "21_Night_shot": "Night shots",
        "22_Stage_concert": "Stage Scenes",
        "23_Fireworks": "Fireworks",
        "24_Candle_light": "Night shots",
        "25_Neon_lights": "Night shots",
        "26_Indoor": "Indoor",
        # "27_Backlight" : "",
        "28_Text_Documents": "Documents",
        "29_QR_images": "QR codes",
        "30_Computer_Screens": "Screenshots"
    }
    dataset_slice_dirs = {
        "training": "train",
        "validation": "val",
        "test": "test"
    }
    zip_dir = os.path.dirname(zip_path)
    unzipped_dir = os.path.join(zip_dir, "CamSDD")
    take_from_ds(unzipped_dir, dataset_slice_dirs, cam_sdd_map)

    # memes ds
    kaggle.api.authenticate()
    ds_name = "n0obcoder/mobile-gallery-image-classification-data"
    kaggle.api.dataset_download_files(ds_name, unzip=True)
    zip_dir = os.getcwd()
    dir_name = "mobile_gallery_image_classification"
    unzipped_dir = os.path.join(zip_dir, dir_name)
    mgicd_map = {
        "Memes": "Memes",
        "Selfies": "Selfies",
        "Trees": "Greenery",
        "Mountains": "Hills"
    }
    src_dir = os.path.join(unzipped_dir, "train")
    dst_dir = os.path.join(gic_dir, "train")
    copy_map_dirs(src_dir, dst_dir, mgicd_map)

    # receipts ds
    ds_url = "https://expressexpense.com/large-receipt-image-dataset-SRD.zip"
    dst_dir = os.path.join(gic_dir, "train", "Receipts")
    ds_fname = os.path.basename(ds_url)
    os.system(f"wget {ds_url} && unzip -qq -d {dst_dir} {ds_fname}")

    # banking cards ds
    ds_id = "1uUCZzLRZKKoc5JLRZiJ34dOXCADeNCtU"
    ds_fname = "banking_cards.tar.xz"
    gdown.download(id=ds_id, output=ds_fname, quiet=True)
    with tarfile.open(ds_fname) as f:
        f.extractall()
    ds_map = {
        "credit_card": "Banking Cards",
    }
    dir_name = "banking_cards"
    take_from_ds(dir_name, dataset_slice_dirs, ds_map)

    # art ds
    ds_name = "thedownhill/art-images-drawings-painting-sculpture-engraving"
    kaggle.api.dataset_download_files(ds_name, unzip=True)
    zip_dir = os.getcwd()
    dir_name = "dataset"
    unzipped_dir = os.path.join(zip_dir, dir_name, "dataset_updated")
    dataset_slice_dirs = {"training_set": "train", "validation_set": "val"}
    ds_map = {
        "drawings": "Art",
        "engraving": "Art",
        "iconography": "Art",
        "painting": "Art",
        "sculpture": "Sculptures"
    }
    take_from_ds(unzipped_dir, dataset_slice_dirs, ds_map)

    # animals ds
    ds_name = "iamsouravbanerjee/animal-image-dataset-90-different-animals"
    kaggle.api.dataset_download_files(ds_name, unzip=True)
    zip_dir = os.getcwd()
    dir_name = "animals"
    unzipped_dir = os.path.join(zip_dir, dir_name, "animals")
    dst_dir = os.path.join(gic_dir, "train")
    ds_map = {
        "antelope": "Animals",
        "badger": "Animals",
        "bat": "Birds",
        "bear": "Animals",
        "bee": "Animals",
        "beetle": "Animals",
        "bison": "Animals",
        "boar": "Animals",
        "butterfly": "Animals",
        "cat": "Cats",
        "caterpillar": "Animals",
        "chimpanzee": "Animals",
        "cockroach": "Animals",
        "cow": "Animals",
        "coyote": "Animals",
        "crab": "Animals",
        "crow": "Birds",
        "deer": "Animals",
        "dog": "Dogs",
        "dolphin": "Animals",
        "donkey": "Animals",
        "dragonfly": "Animals",
        "duck": "Birds",
        "eagle": "Birds",
        "elephant": "Animals",
        "flamingo": "Birds",
        "fly": "Animals",
        "fox": "Animals",
        "goat": "Animals",
        "goldfish": "Animals",
        "goose": "Birds",
        "gorilla": "Animals",
        "grasshopper": "Animals",
        "hamster": "Animals",
        "hare": "Animals",
        "hedgehog": "Animals",
        "hippopotamus": "Animals",
        "hornbill": "Birds",
        "horse": "Animals",
        "hummingbird": "Birds",
        "hyena": "Animals",
        "jellyfish": "Animals",
        "kangaroo": "Animals",
        "koala": "Animals",
        "ladybugs": "Animals",
        "leopard": "Animals",
        "lion": "Animals",
        "lizard": "Animals",
        "lobster": "Animals",
        "mosquito": "Animals",
        "moth": "Animals",
        "mouse": "Animals",
        "octopus": "Animals",
        "okapi": "Animals",
        "orangutan": "Animals",
        "otter": "Animals",
        "owl": "Birds",
        "ox": "Animals",
        "oyster": "Animals",
        "panda": "Animals",
        "parrot": "Birds",
        "pelecaniformes": "Birds",
        "penguin": "Birds",
        "pig": "Animals",
        "pigeon": "Birds",
        "porcupine": "Animals",
        "possum": "Animals",
        "raccoon": "Animals",
        "rat": "Animals",
        "reindeer": "Animals",
        "rhinoceros": "Animals",
        "sandpiper": "Animals",
        "seahorse": "Animals",
        "seal": "Animals",
        "shark": "Animals",
        "sheep": "Animals",
        "snake": "Animals",
        "sparrow": "Birds",
        "squid": "Animals",
        "squirrel": "Birds",
        "starfish": "Animals",
        "swan": "Birds",
        "tiger": "Animals",
        "turkey": "Birds",
        "turtle": "Animals",
        "whale": "Animals",
        "wolf": "Animals",
        "wombat": "Animals",
        "woodpecker": "Birds",
        "zebra": "Animals"
    }
    copy_map_dirs(unzipped_dir, dst_dir, ds_map)

    # oi ds
    dst_dir = os.path.join(os.getcwd(), "oi")
    oi_c_label_map = {
        "Baby": "Babies",
        "Hindu temple": "God",
        "Ganesh chaturthi": "Ganesh",
        "Fruit": "Fruits",
        "Wine": "Wine",
        "Desert": "Deserts",
        "Presentation": "Presentations",
        "Dining room": "Dining",
        "Pub": "Pub",
        "Stadium": "Stadium",
        "Gym": "Gym",
        "Grocery store": "Stores",
        "Shopping mall": "Shopping mall",
        "Body jewelry": "Jewelry",
        "Electronic device": "Electronics",
        "Mobile phone": "Cell Phones"
    }
    src_dirs_map = oi_download.download_images(
        dest_dir=dst_dir,
        class_labels=oi_c_label_map.keys(),
        exclusions_path=None,
        meta_dir=dst_dir,
        limit=500)
    take_from_oi(oi_c_label_map, src_dirs_map)

    # clothes ds
    ds_name = "paramaggarwal/fashion-product-images-small"
    kaggle.api.dataset_download_files(ds_name, unzip=True)
    zip_dir = os.getcwd()
    dir_name = "myntradataset"
    unzipped_dir = os.path.join(zip_dir, dir_name, "images")
    dst_dir = os.path.join(gic_dir, "train", "Clothes")
    shutil.move(unzipped_dir, dst_dir)

    os.chdir(os.path.dirname(ims_dir))
    shutil.rmtree(ims_dir)

    train_dir = os.path.join(gic_dir, "train")
    rm_count = 0
    valid_hdrs = ("jpg", "jpeg", "png", "gif", "bmp")

    for dir_name in os.listdir(train_dir):
        cat_dir_path = os.path.join(train_dir, dir_name)

        for img in os.listdir(cat_dir_path):
            img_path = os.path.join(cat_dir_path, img)
            f_size = (os.path.getsize(img_path) == 0)
            img_hdr = imghdr.what(img_path)

            if (f_size or (img_hdr not in valid_hdrs)):
                os.remove(img_path)
                rm_count += 1
    print(f"Removed {rm_count} files.")


def main():
    """Prepare dataset and exit."""
    prep_ds()


if __name__ == "__main__":
    main()

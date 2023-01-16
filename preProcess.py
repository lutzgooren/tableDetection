import json
import os
from PIL import Image, ImageDraw
import pandas as pd
import numpy as np

"""
struktur des ordners:
root
|_____data
        |_____*.jpg
|_____labels.json
"""

# setze pfade
img_dir = os.path.join("stuff", "data")
annot_file = os.path.join("stuff", "labels.json")


def extract_data():
    """
    lade daten und gib annotations, images und filtered annotations als listen zurück
    id der kategorien gehören zu folgenden labeln:
    1 - ueberschrift
    2 - tabelle
    3 - kopfzeile
    4 - vorspalte
    5 - felder
    6 - spalte
    """

    # öffne annotation datei, lade daten in ein json objekt und schließe anschließend den filestream
    f = open(annot_file)
    json_data = json.load(f)
    f.close()

    # hole image und annotation informationen aus dem json objekt und speichere diese in eigenen listen
    j_annot = json_data["annotations"]
    j_imgs = json_data["images"]

    # erstelle eigene liste für jedes label
    hl_list = []
    t_list = []
    th_list = []
    fc_list = []
    f_list = []
    c_list = []

    # füge annotation objekt zur passenden liste hinzu
    for annot_object in j_annot:
        match annot_object["category_id"]:
            case 1:
                hl_list.append(annot_object)
            case 2:
                t_list.append(annot_object)
            case 3:
                th_list.append(annot_object)
            case 4:
                fc_list.append(annot_object)
            case 5:
                f_list.append(annot_object)
            case 6:
                c_list.append(annot_object)
    return j_imgs, j_imgs, hl_list, t_list, th_list, fc_list, f_list, c_list


def retrieve_image_path(id, imgs):
    """
    funktion um einen dateinamen mit hilfe einer id aus einer liste zu finden
    wird benötigt, damit das neuronale netz die später generierten masken zu den dateinamen zuordnen kann
    """
    for image in imgs:
        if image["id"] == id:
            return image["file_name"]
    return -1


# hole alle benötigten informationen mit der extract_data() funktion aus der annotation datei (.json)
json_annot, \
    json_imgs, \
    headline_list, \
    table_list, \
    tablehead_list, \
    frontcolumn_list, \
    field_list, \
    column_list = extract_data()


def get_bboxes(headlines, tables, tableheads, frontcolumns, fields, columns, verbose=False):
    """
    hole bounding boxes aus den annotations der verschiedenen klassen
    bounding boxes werden als 2 koordinaten gespeichert ([x, y], [x, y]) zwischen denen ein rechteck gespannt wird
    die rechtecke repräsentieren den umfang, in welchem eine gewisse information zu sehen ist (z.B. tabelle/spalte)
    die funktion speichert die koordinaten in einem bounding box dictionary mit folgender struktur:

    b_dict{}
    |_____[filename]
                |_____headlines
                        |_____list([[bounding_box_coordinates],[...]])
                |_____tables
                        |_____list([[bounding_box_coordinates],[...]])
                |_____tableheads
                        |_____list([[bounding_box_coordinates],[...]])
                |_____frontcolumns
                        |_____list([[bounding_box_coordinates],[...]])
                |_____fields
                        |_____list([[bounding_box_coordinates],[...]])
                |_____columns
                        |_____list([[bounding_box_coordinates],[...]])
    |_____[filename]
    [...]
    """
    b_dict = {}
    for u in headlines:
        fname = retrieve_image_path(u["image_id"], json_imgs)
        if fname in b_dict:
            if verbose:
                print("already exists")
        else:
            b_dict[fname] = dict()
        if "headlines" in b_dict[fname]:
            if verbose:
                print("already exists")
        else:
            b_dict[fname]["headlines"] = list()
        b_dict[fname]["headlines"].append(u["bbox"])

    for v in tables:
        fname = retrieve_image_path(v["image_id"], json_imgs)
        if fname in b_dict:
            if verbose:
                print("already exists")
        else:
            b_dict[fname] = dict()
        if "tables" in b_dict[fname]:
            if verbose:
                print("already exists")
        else:
            b_dict[fname]["tables"] = list()
        b_dict[fname]["tables"].append(v["bbox"])

    for w in tableheads:
        fname = retrieve_image_path(w["image_id"], json_imgs)
        if fname in b_dict:
            if verbose:
                print("already exists")
        else:
            b_dict[fname] = dict()
        if "tableheads" in b_dict[fname]:
            if verbose:
                print("already exists")
        else:
            b_dict[fname]["tableheads"] = list()
        b_dict[fname]["tableheads"].append(w["bbox"])

    for x in frontcolumns:
        fname = retrieve_image_path(x["image_id"], json_imgs)
        if fname in b_dict:
            if verbose:
                print("already exists")
        else:
            b_dict[fname] = dict()
        if "frontcolumns" in b_dict[fname]:
            if verbose:
                print("already exists")
        else:
            b_dict[fname]["frontcolumns"] = list()
        b_dict[fname]["frontcolumns"].append(x["bbox"])

    for y in fields:
        fname = retrieve_image_path(y["image_id"], json_imgs)
        if fname in b_dict:
            if verbose:
                print("already exists")
        else:
            b_dict[fname] = dict()
        if "fields" in b_dict[fname]:
            if verbose:
                print("already exists")
        else:
            b_dict[fname]["fields"] = list()
        b_dict[fname]["fields"].append(y["bbox"])

    for z in columns:
        fname = retrieve_image_path(z["image_id"], json_imgs)
        if fname in b_dict:
            if verbose:
                print("already exists")
        else:
            b_dict[fname] = dict()
        if "columns" in b_dict[fname]:
            if verbose:
                print("already exists")
        else:
            b_dict[fname]["columns"] = list()
        b_dict[fname]["columns"].append(z["bbox"])
    return b_dict


# speichere bounding boxes in bbox dictionary
bbox_dict = get_bboxes(headline_list, table_list, tablehead_list, frontcolumn_list, field_list, column_list)


def save_masks(fname, list_bbox, dim, type):
    """
    erstellt masken aus bounding box liste und speichert diese in einer .jpg datei ab
    masken werden auf typbasis erstellt - masken für die spalten werden z.B. in masked/columns gespeichert usw.
    """
    mask_dir = os.path.join("stuff", "masked", type)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    image = Image.new("RGB", dim)
    mask = ImageDraw.Draw(image)
    for each_list in list_bbox:
        [x, y, width, height] = each_list
        mask.rectangle([x, y, x + width, y + height], fill=255)
    mask_fname = os.path.join(mask_dir, fname)
    image = np.array(image)
    image = Image.fromarray(image[:, :, 0])
    image.save(mask_fname)
    return mask_fname


def create_masks_df():
    """
    nutze alles bisher gesammelte, um einen dataframe zu erstellen, sodass alle informationen an einem ort sind
    spalten beinhalten den pfad des bildes und alle zugehörigen pfade der masken
    """
    img_df = pd.DataFrame(
        columns=["image_path", "headlinemask_path", "tablemask_path", "tableheadmask_path",
                 "frontcolumnmask_path", "fieldmask_path", "columnmask_path"])
    csv_dir = os.path.join("stuff", "csv")
    if not os.path.exists(csv_dir):
        os.mkdir(csv_dir)
    for file_name, bboxes_dict in bbox_dict.items():
        img_path = os.path.join(img_dir, file_name)
        dim = Image.open(img_path).size
        for type, bboxes, in bboxes_dict.items():
            match type:
                case "headlines":
                    headlinemask_path = save_masks(file_name, bboxes, dim, type)
                case "tables":
                    tablemask_path = save_masks(file_name, bboxes, dim, type)
                case "tableheads":
                    tableheadmask_path = save_masks(file_name, bboxes, dim, type)
                case "frontcolumns":
                    frontcolumnmask_path = save_masks(file_name, bboxes, dim, type)
                case "fields":
                    fieldmask_path = save_masks(file_name, bboxes, dim, type)
                case "columns":
                    columnmask_path = save_masks(file_name, bboxes, dim, type)
        # füge alle gesammelten informationen aus dem derzeitigen schleifendurchlauf am ende des dataframes ein
        img_df.loc[len(img_df.index)] = [img_path, headlinemask_path, tablemask_path, tableheadmask_path,
                                         frontcolumnmask_path, fieldmask_path, columnmask_path]
    # speichere den dataframe in einer .csv datei für die spätere nutzung mit dem neuronalen netz
    csv_fname = os.path.join(csv_dir, "data.csv")
    img_df.to_csv(csv_fname, index=False)
    return img_df


"""
create_masks_df() muss nur einmal ausgeführt werden, um die masken und die .csv für das netz zu erstellen
erneut ausführen, falls neue trainingsdaten hinzugefügt werden sollen
muss zu keiner variable zugewiesen werden - die funktion würde auch ohne ein return funktionieren
nur für den fall, dass ich mich dazu entscheide die variable für debugging zwecke zu untersuchen oder falls ich die
methode aus einem anderen python script aufrufen möchte
"""
img_df = create_masks_df()
print(img_df)

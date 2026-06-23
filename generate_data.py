import pandas as pd
import random

random.seed(42)

silhouettes = [
    "A-Line", "Fit and Flare", "Straight", "Bodycon",
    "Shift", "Empire", "Peplum", "Wrap"
]

fabrics = [
    "Cotton", "Polyester", "Rayon", "Linen",
    "Silk", "Georgette", "Chiffon", "Denim"
]

necklines = [
    "Round Neck", "V-Neck", "Boat Neck", "Square Neck",
    "Collared", "Sweetheart Neck", "Halter Neck", "Keyhole Neck"
]

sleeves = [
    "Sleeveless", "Short Sleeve", "Half Sleeve",
    "Three-Quarter Sleeve", "Full Sleeve", "Puff Sleeve"
]

lengths = [
    "Mini", "Knee Length", "Midi", "Maxi"
]

embellishments = [
    "None", "Sequins", "Embroidery",
    "Lace", "Beads", "Printed", "Ruffles", "Applique"
]

colors = [
    "Black", "White", "Red", "Blue",
    "Green", "Pink", "Yellow", "Purple"
]

categories = [
    "Dress", "Top", "Kurti", "Shirt",
    "Gown", "Tunic", "Jumpsuit", "Co-ord Set"
]

rows = []

for i in range(200):

    silhouette = random.choice(silhouettes)
    fabric = random.choice(fabrics)
    neckline = random.choice(necklines)
    sleeve = random.choice(sleeves)
    length = random.choice(lengths)
    embellishment = random.choice(embellishments)
    color = random.choice(colors)
    category = random.choice(categories)

    description = (
        f"Elegant {color.lower()} {category.lower()} crafted from "
        f"{fabric.lower()} fabric featuring a {neckline.lower()}, "
        f"{sleeve.lower()}, {length.lower()} length and "
        f"{embellishment.lower()} detailing. Designed in a "
        f"{silhouette.lower()} silhouette for a stylish look."
    )

    rows.append([
        description,
        silhouette,
        fabric,
        neckline,
        sleeve,
        length,
        embellishment,
        color,
        category
    ])

df = pd.DataFrame(rows, columns=[
    "description",
    "silhouette",
    "fabric",
    "neckline",
    "sleeve",
    "length",
    "embellishment",
    "color",
    "category"
])

df.to_csv("dataset.csv", index=False)

print("CSV saved successfully with", len(df), "rows")
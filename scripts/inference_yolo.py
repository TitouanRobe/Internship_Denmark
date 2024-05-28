from ultralytics import YOLO
from PIL import Image
import requests

model = YOLO("./models/best.engine")


url = "https://c8.alamy.com/comp/2ET5JP5/damaged-road-sign-stop-covered-with-scratches-and-rusty-rumpled-road-sign-on-a-blue-sky-background-stop-sign-with-partly-bent-surface-urban-grunge-2ET5JP5.jpg"
#url = "https://habitatbyresene.co.nz/assets/Uploads/1__FillMaxWzgwMCw2MDBd_WatermarkWzEwLDEwLDI0MCw3MiwiQm90dG9tUmlnaHQiLDIyMDg3XQ.png"
#url = "https://media.defense.gov/2010/Aug/06/2000337077/2000/2000/0/100805-F-2003B-001.JPG"
#url = "https://as1.ftcdn.net/v2/jpg/04/51/13/26/1000_F_451132633_kzbaCfrK3svyU1QhoPPzJukJqKzJvYLq.jpg"
#url = "https://www.shutterstock.com/shutterstock/photos/89764/display_1500/stock-photo-old-rusty-stop-sign-89764.jpg"
url = "https://facts.net/wp-content/uploads/2023/12/11-stop-sign-facts-1701615226.jpg"
url = "https://i.dailymail.co.uk/1s/2018/10/10/17/4928442-0-image-a-76_1539189734487.jpg"
url ="https://bikeportland.org/wp-content/uploads/2018/07/IMG_4177-e1530815628761-scaled.jpg"
#url="https://www.magbloom.com/wp-content/uploads/2018/09/ROTATOR.jpg"
url = "https://www.telegraph.co.uk/content/dam/news/2024/03/12/TELEMMGLPICT000356897456_17102801404780_trans_NvBQzQNjv4BqrpfQw2hJyG_yckwxPAr0gmyy-GsNrhPQbLesooHneHs.jpeg"
url = "https://as1.ftcdn.net/v2/jpg/04/51/13/26/1000_F_451132633_kzbaCfrK3svyU1QhoPPzJukJqKzJvYLq.jpg"


url = "https://www.shutterstock.com/image-photo/08202023-crete-greece-damaged-falling-260nw-2375880711.jpg"


image = Image.open(requests.get(url, stream=True).raw)
# image.show()

image = "./input/give_way.jpg"

results = model.predict(image, conf=0.5, save=False)


# View results
for r in results:
    print(r.masks)  # print the Masks object containing the detected instance masks
    print(r.boxes)

# Visualize the results
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()


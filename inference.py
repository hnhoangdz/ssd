from cfg import *
from models.ssd import SSD
from data.transform import DataTransform
import torch
import cv2
import matplotlib.pyplot as plt

net = SSD(phase="inference", cfg=cfg)
net_weights = torch.load("weights/ssd300_100.pth", map_location={"cuda:0":"cpu"})
net.load_state_dict(net_weights)

def show_predict(img_file_path):
    img = cv2.imread(img_file_path)
    transform = DataTransform(input_size, color_mean)

    phase = "val"
    img_tranformed, boxes, labels = transform(img, phase, "", "")
    img_tensor = torch.from_numpy(img_tranformed[:,:,(2,1,0)]).permute(2,0,1)

    net.eval()
    input = img_tensor.unsqueeze(0) #(1, 3, 300, 300)
    output = net(input)

    plt.figure(figsize=(10, 10))
    colors = [(255,0,0), (0,255,0), (0,0,255)]
    font = cv2.FONT_HERSHEY_SIMPLEX

    detections = output.data #(1, 21, 200, 5) 5: score, cx, cy, w, h
    scale = torch.Tensor(img.shape[1::-1]).repeat(2)

    for i in range(detections.size(1)):
        print(i)
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            score = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            print('Bounding boxes: ', pt)
            cv2.rectangle(img,
                        (int(pt[0]), int(pt[1])),
                        (int(pt[2]), int(pt[3])),
                        colors[i%3], 2
                        )
            display_text = "%s: %.2f"%(class_names[i-1], score)
            cv2.putText(img, display_text, (int(pt[0]), int(pt[1])),
                font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            j += 1
    
    cv2.imshow("Result", img)
    cv2.imwrite('results/result.jpg', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    file_path = 'images/cowboy.jpg'
    show_predict(file_path)
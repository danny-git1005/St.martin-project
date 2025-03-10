import cv2
import numpy as np
import onnxruntime as ort

class yolov6():
    def __init__(self, modelpath, confThreshold=0.5, nmsThreshold=0.5):
        # self.classes = list(map(lambda x:x.strip(), open('coco.names', 'r').readlines()))
        # self.num_classes = len(self.classes)
        self.inpHeight, self.inpWidth = 320, 320
        so = ort.SessionOptions()
        so.log_severity_level = 3
        
        provider = ['CPUExecutionProvider']
        # provider = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        self.net = ort.InferenceSession(modelpath, so, provider)
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.keep_ratio=True

    def resize_image(self, srcimg):
        top, left, newh, neww = 0, 0, self.inpWidth, self.inpHeight
        if self.keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.inpHeight, int(self.inpWidth / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.inpWidth - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.inpWidth - neww - left, cv2.BORDER_CONSTANT,
                                         value=(114, 114, 114))  # add border
            else:
                newh, neww = int(self.inpHeight * hw_scale), self.inpWidth
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.inpHeight - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.inpHeight - newh - top, 0, 0, cv2.BORDER_CONSTANT,
                                         value=(114, 114, 114))
        else:
            img = cv2.resize(srcimg, (self.inpWidth, self.inpHeight), interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return img

    def postprocess(self, frame, outs, padsize=None):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        newh, neww, padh, padw = padsize
        ratioh, ratiow = frameHeight / newh, frameWidth / neww
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.

        confidences = []
        boxes = []
        temp = []
        classIds = []
        for detection in outs:
            if detection[4] > self.confThreshold:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId] * detection[4]
                if confidence > self.confThreshold:
                    center_x = int((detection[0] - padw) * ratiow)
                    center_y = int((detection[1] - padh) * ratioh)
                    width = int(detection[2] * ratiow)
                    height = int(detection[3] * ratioh)
                    left = int(center_x - width * 0.5)
                    top = int(center_y - height * 0.5)

                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
                    classIds.append(classId)
        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        bbox = []
        if len(boxes) != 0:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
            indices = indices.flatten()
            
            for i in indices:
                box = boxes[i]
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]
                conf = confidences[i]
                classid = classIds[i]
                bbox.append( [left,top,width,height, conf, classid] )
                frame = self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)

        return frame, bbox

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=1)

        label = '%.2f' % conf
        # label = '%s:%s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=1)
        return frame

    def detect(self, srcimg):
        img, newh, neww, padh, padw = self.resize_image(srcimg)
        img = self.preprocess(img)
        # Sets the input to the network
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

        outs = self.net.run(None, {self.net.get_inputs()[0].name: blob})[0].squeeze(axis=0)

        srcimg = self.postprocess(srcimg, outs, padsize=(newh, neww, padh, padw))
        return srcimg

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--imgpath', type=str, default='3.jpg', help="image path")
#     parser.add_argument('--modelpath', type=str, default='best_ckpt.onnx')
#     parser.add_argument('--confThreshold', default=0.3, type=float, help='class confidence')
#     parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
#     args = parser.parse_args()

#     yolonet = yolov6(args.modelpath, confThreshold=args.confThreshold, nmsThreshold=args.nmsThreshold)
#     srcimg = cv2.imread(args.imgpath)
#     srcimg,bbox = yolonet.detect(srcimg)

#     winName = 'Deep learning object detection in ONNXRuntime'
#     cv2.namedWindow(winName, 0)
#     cv2.imshow(winName, srcimg)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# https://github.com/hpc203/yolov6-opencv-onnxruntime/blob/main/onnxruntime/main.py
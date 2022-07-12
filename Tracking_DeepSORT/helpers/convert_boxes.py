def convert_boxes(image, boxes): # expects tlbr bboxes
    returned_boxes = []
    for box in boxes:
        box = box.astype(int)
        box = box.tolist()
        if box != [0,0,0,0]:
            returned_boxes.append(box)
    return returned_boxes
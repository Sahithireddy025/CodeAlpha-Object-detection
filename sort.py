import numpy as np

class Sort:
    def __init__(self):
        self.next_id = 1
        self.objects = {}

    def update(self, detections):
        updated_objects = []

        if len(detections) == 0:
            return np.empty((0, 5))

        for det in detections:
            x1, y1, x2, y2, score = det

            # Assign new ID
            obj_id = self.next_id
            self.next_id += 1

            self.objects[obj_id] = [x1, y1, x2, y2]

            updated_objects.append([x1, y1, x2, y2, obj_id])

        return np.array(updated_objects)
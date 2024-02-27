def iou_score(bbox1, bbox2):
    """Jaccard index or Intersection over Union.

    https://en.wikipedia.org/wiki/Jaccard_index

    bbox: [xmin, ymin, xmax, ymax]
    """

    assert len(bbox1) == 4
    assert len(bbox2) == 4

    # Write code here
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2

    x_right = min(xmax1, xmax2)
    y_bottom = min(ymax1, ymax2)
    x_left = max(xmin1, xmin2)
    y_top = max(ymin1, ymin2)

    if x_right - x_left < 0 or y_bottom - y_top < 0:
        return 0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    union = (xmax1 - xmin1) * (ymax1 - ymin1) + (xmax2 - xmin2) * (ymax2 - ymin2) - intersection

    return intersection / union


def motp(obj, hyp, threshold=0.5):
    """Calculate MOTP

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Write code here

        # Step 1: Convert frame detections to dict with IDs as keys
        frame_obj_dict = {frame_obj[i][0]: frame_obj[i][1:] for i in range(len(frame_obj))}
        frame_hyp_dict = {frame_hyp[i][0]: frame_hyp[i][1:] for i in range(len(frame_hyp))}

        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        for (obj_id, hyp_id) in matches.keys():
            if obj_id in frame_obj_dict.keys() and hyp_id in frame_hyp_dict.keys():
                iou = iou_score(frame_obj_dict[obj_id], frame_hyp_dict[hyp_id])
                if iou > threshold:
                    match_count += 1
                    dist_sum += iou
                    frame_obj_dict.pop(obj_id)
                    frame_hyp_dict.pop(hyp_id)


        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        current_matches = []

        for obj_id in frame_obj_dict.keys():
            for hyp_id in frame_hyp_dict.keys():
                iou = iou_score(frame_obj_dict[obj_id], frame_hyp_dict[hyp_id])
                if iou > threshold:
                    current_matches.append([iou, obj_id, hyp_id])

        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        current_matches = sorted(current_matches, key=lambda x: x[0], reverse=True)

        for iou, obj_id, hyp_id in current_matches:
            if obj_id in frame_obj_dict.keys():
                frame_obj_dict.pop(obj_id)
            else:
                continue
            if hyp_id in frame_hyp_dict.keys():
                frame_hyp_dict.pop(hyp_id)
            else:
                continue

            match_count += 1
            dist_sum += iou
            matches[(obj_id, hyp_id)] = iou

        # Step 5: Update matches with current matched IDs
        matches = {}
        for iou, obj_id, hyp_id in current_matches:
            matches[(obj_id, hyp_id)] = iou

    # Step 6: Calculate MOTP
    MOTP = dist_sum / match_count

    return MOTP


def motp_mota(obj, hyp, threshold=0.4):
    """Calculate MOTP/MOTA

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0
    missed_count = 0
    false_positive = 0
    mismatch_error = 0
    obj_num = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for t, (frame_obj, frame_hyp) in enumerate(zip(obj, hyp)):
        obj_num += len(frame_obj)

        # Step 1: Convert frame detections to dict with IDs as keys
        frame_obj_dict = {frame_obj[i][0]: frame_obj[i][1:] for i in range(len(frame_obj))}
        frame_hyp_dict = {frame_hyp[i][0]: frame_hyp[i][1:] for i in range(len(frame_hyp))}

        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        for obj_id, hyp_id in matches.items():
            if obj_id in frame_obj_dict.keys() and hyp_id in frame_hyp_dict.keys():
                iou = iou_score(frame_obj_dict[obj_id], frame_hyp_dict[hyp_id])
                if iou > threshold:
                    match_count += 1
                    dist_sum += iou
                    frame_obj_dict.pop(obj_id)
                    frame_hyp_dict.pop(hyp_id)

        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        current_matches = []

        for obj_id in frame_obj_dict.keys():
            for hyp_id in frame_hyp_dict.keys():
                iou = iou_score(frame_obj_dict[obj_id], frame_hyp_dict[hyp_id])
                if iou > threshold:
                    current_matches.append([iou, obj_id, hyp_id])

        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections

        # Step 5: If matched IDs contradict previous matched IDs - increase mismatch error

        # Step 6: Update matches with current matched IDs
        current_matches = sorted(current_matches, key=lambda x: x[0], reverse=True)

        for iou, obj_id, hyp_id in current_matches:
            if obj_id in frame_obj_dict:
                dist_sum += iou
                match_count += 1
                if obj_id not in matches:
                    for prev_obj, prev_hyp_id in matches.items():
                        if prev_hyp_id == hyp_id:
                            matches.pop(prev_obj)
                            break
                    matches[obj_id] = hyp_id
                elif matches[obj_id] != hyp_id:
                    matches.pop(obj_id)
                    matches[obj_id] = hyp_id
                    mismatch_error += 1
                if obj_id in frame_obj_dict.keys():
                    frame_obj_dict.pop(obj_id)
                if hyp_id in frame_hyp_dict.keys():
                    frame_hyp_dict.pop(hyp_id)

        # Step 7: Errors
        # All remaining hypotheses are considered false positives
        # All remaining objects are considered misses
        false_positive += len(list(frame_hyp_dict.keys()))
        missed_count += len(list(frame_obj_dict.keys()))

    # Step 8: Calculate MOTP and MOTA
    MOTP = dist_sum / (match_count + 1e-8)
    MOTA = 1 - (false_positive + missed_count + mismatch_error) / (obj_num + 1e-8)

    return MOTP, MOTA

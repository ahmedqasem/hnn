import cv2
import matplotlib.pyplot as plt
import numpy as np


def draw_sample(C, X, Y, image_data, debug_img, debug_num_pos):

    if debug_num_pos == 0:
        gt_x1, gt_x2 = image_data['bboxes'][0]['x1'] * (X.shape[2] / image_data['height']), image_data['bboxes'][0][
            'x2'] * (X.shape[2] / image_data['height'])
        gt_y1, gt_y2 = image_data['bboxes'][0]['y1'] * (X.shape[1] / image_data['width']), image_data['bboxes'][0][
            'y2'] * (X.shape[1] / image_data['width'])
        gt_x1, gt_y1, gt_x2, gt_y2 = int(gt_x1), int(gt_y1), int(gt_x2), int(gt_y2)

        img = debug_img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        color = (0, 255, 0)
        cv2.putText(img, 'gt bbox', (gt_x1, gt_y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)
        cv2.rectangle(img, (gt_x1, gt_y1), (gt_x2, gt_y2), color, 2)
        cv2.circle(img, (int((gt_x1 + gt_x2) / 2), int((gt_y1 + gt_y2) / 2)), 3, color, 0.5)

        plt.grid()
        plt.imshow(img)
        plt.show()
    else:
        cls = Y[0][0]
        pos_cls = np.where(cls == 1)
        print(pos_cls)
        regr = Y[1][0]
        pos_regr = np.where(regr == 1)
        print(pos_regr)
        print('y_rpn_cls for possible pos anchor: {}'.format(cls[pos_cls[0][0], pos_cls[1][0], :]))
        print('y_rpn_regr for positive anchor: {}'.format(regr[pos_regr[0][0], pos_regr[1][0], :]))

        gt_x1, gt_x2 = image_data['bboxes'][0]['x1'] * (X.shape[2] / image_data['width']), image_data['bboxes'][0][
            'x2'] * (X.shape[2] / image_data['width'])
        gt_y1, gt_y2 = image_data['bboxes'][0]['y1'] * (X.shape[1] / image_data['height']), image_data['bboxes'][0][
            'y2'] * (X.shape[1] / image_data['height'])
        gt_x1, gt_y1, gt_x2, gt_y2 = int(gt_x1), int(gt_y1), int(gt_x2), int(gt_y2)

        img = debug_img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        color = (0, 255, 0)
        #   cv2.putText(img, 'gt bbox', (gt_x1, gt_y1-5), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)
        cv2.rectangle(img, (gt_x1, gt_y1), (gt_x2, gt_y2), color, 2)
        cv2.circle(img, (int((gt_x1 + gt_x2) / 2), int((gt_y1 + gt_y2) / 2)), 3, color, -1)

        # Add text
        textLabel = 'gt bbox'
        (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
        textOrg = (gt_x1, gt_y1 + 5)
        cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                      (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 2)
        cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                      (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
        cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

        # Draw positive anchors according to the y_rpn_regr
        for i in range(debug_num_pos):
            color = (100 + i * (155 / 4), 0, 100 + i * (155 / 4))

            idx = pos_regr[2][i * 4] / 4
            anchor_size = C.anchor_box_scales[int(idx / 3)]
            anchor_ratio = C.anchor_box_ratios[2 - int((idx + 1) % 3)]

            center = (pos_regr[1][i * 4] * C.rpn_stride, pos_regr[0][i * 4] * C.rpn_stride)
            print('Center position of positive anchor: ', center)
            cv2.circle(img, center, 3, color, -1)
            anc_w, anc_h = anchor_size * anchor_ratio[0], anchor_size * anchor_ratio[1]
            cv2.rectangle(img, (center[0] - int(anc_w / 2), center[1] - int(anc_h / 2)),
                          (center[0] + int(anc_w / 2), center[1] + int(anc_h / 2)), color, 2)
    #         cv2.putText(img, 'pos anchor bbox '+str(i+1), (center[0]-int(anc_w/2), center[1]-int(anc_h/2)-5), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

    print('Green bboxes is ground-truth bbox. Others are positive anchors')
    plt.figure(figsize=(8, 8))
    plt.grid()
    plt.imshow(img)
    plt.show()


def show_tarining_history(record_df):
    r_epochs = len(record_df)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, r_epochs), record_df['mean_overlapping_bboxes'], 'r')
    plt.title('mean_overlapping_bboxes')

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, r_epochs), record_df['class_acc'], 'r')
    plt.title('class_acc')

    plt.show()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, r_epochs), record_df['loss_rpn_cls'], 'r')
    plt.title('loss_rpn_cls')

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, r_epochs), record_df['loss_rpn_regr'], 'r')
    plt.title('loss_rpn_regr')
    plt.show()
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, r_epochs), record_df['loss_class_cls'], 'r')
    plt.title('loss_class_cls')

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, r_epochs), record_df['loss_class_regr'], 'r')
    plt.title('loss_class_regr')
    plt.show()
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, r_epochs), record_df['curr_loss'], 'r')
    plt.title('total_loss')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, r_epochs), record_df['elapsed_time'], 'r')
    plt.title('elapsed_time')

    plt.show()
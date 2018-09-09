#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Scanned Photo Splitter

Split and adjust separate photos in a scanned image.
"""

import argparse
import itertools
import json
import logging
import math
import sys

import cv2
import numpy
import shapely.geometry

SETTINGS_PATH = './split_settings.json'

KEY_ENTER = 13
KEY_ESCAPE = 27
KEY_LEFT = 81
KEY_UP = 82
KEY_RIGHT = 83
KEY_DOWN = 84
KEY_PAGE_UP = 85
KEY_PAGE_DOWN = 86


def main():
  arg_parser = argparse.ArgumentParser()
  arg_parser.description = __doc__
  arg_parser.formatter_class = argparse.RawDescriptionHelpFormatter
  arg_parser.add_argument('--debug', help='Debug level logging')
  arg_parser.add_argument('image_list', nargs='+', help='Path to scanned image')
  args = vars(arg_parser.parse_args())

  logging.basicConfig(level=logging.DEBUG if args['debug'] else logging.INFO)

  param_dict = load_params()

  cv2.namedWindow('Scanned Photo Splitter')
  # cv2.moveWindow('result', 2000, 100)

  param_dict['image_idx'] = 0
  loaded_image_idx = -1
  orig_rgb = None
  resized_rgb = None

  while True:
    logging.debug('#' * 100)

    print(args['image_list'][param_dict['image_idx']])
    print('Working...')

    if loaded_image_idx != param_dict['image_idx']:
      orig_rgb = cv2.imread(args['image_list'][param_dict['image_idx']])

      resized_rgb = cv2.resize(
        orig_rgb,
        None,
        fx=param_dict['resize_calc_ratio'],
        fy=param_dict['resize_calc_ratio'],
        # Also tried INTER_LANCZOS4 but INTER_AREA works much better. It samples
        # all the pixels, so removes noise in the background.
        interpolation=cv2.INTER_AREA,
      )
      loaded_image_idx = param_dict['image_idx']

    # visualize = image to annotate for display when setting parameters or debugging
    # input = image to use when creating mask
    # mask = 1 8-bit channel
    # rgb = 3 8-bit channels

    background_visualize_rgb = resized_rgb.copy()
    # background_mask = calc_auto_background_mask(resized_rgb, background_visualize_rgb)
    background_mask = calc_manual_tolerance_mask(
      resized_rgb,
      background_visualize_rgb,
      param_dict['flood_fill_tolerance'],
      sample_step=50,
    )

    photo_visualize_rgb = resized_rgb.copy()
    box_list = find_photos(background_mask, photo_visualize_rgb, param_dict)

    box_list = filter_boxes_by_aspect(box_list, min_aspect=0.3)
    box_list = filter_boxes_by_size(box_list, min_area=500)
    box_list = remove_intersecting_smaller_boxes(box_list)

    box_filter_viz_rgb = resized_rgb.copy()

    draw_boxes(box_filter_viz_rgb, box_list, (0, 0, 255))

    h, w = resized_rgb.shape[:2]

    tiled_rgb = create_tiled_image([
      background_visualize_rgb,
      box_filter_viz_rgb,
    ], 2, 2 * w, 1 * h)

    cv2.imshow('result', tiled_rgb)

    print('Ready')

    k = cv2.waitKey(0) & 0xFF

    print(k)

    if k == KEY_ESCAPE:
      break

    if k == KEY_ENTER:
      save_photo_list(orig_rgb, box_list, param_dict)

    if k == ord('s'):
      save_params(param_dict)
      logging.info('Saved parameters')

    adjust(
      k, param_dict, 'image_idx', KEY_UP, KEY_DOWN, max_val=len(args['image_list']) - 1
    )
    adjust(k, param_dict, 'flood_fill_tolerance', KEY_LEFT, KEY_RIGHT, max_val=255)

  cv2.destroyAllWindows()


def adjust(
    key_str, param_dict, param_key, down_key, up_key, min_val=0, max_val=None
):
  # if key_str not in (ord(down_key), ord(up_key)):
  if key_str not in (down_key, up_key):
      return
  if key_str == up_key:
    param_dict[param_key] += 1
  else:
    if param_dict[param_key] > 0:
      param_dict[param_key] -= 1
  if param_dict[param_key] <= min_val:
    param_dict[param_key] = 0
  if max_val is not None and param_dict[param_key] >= max_val:
    param_dict[param_key] = max_val
  print('{}: {}'.format(param_key, param_dict[param_key]))


def save_photo_list(input_rgb, box_list, param_dict):
  box_idx = 0
  prev_box_idx = -1
  photo_rgb = None

  while True:
    if prev_box_idx != box_idx:
      prev_box_idx = box_idx
      photo_rgb = extract_and_rotate_photo(
        input_rgb,
        numpy.array(
          scale_box(
            box_list[box_idx],
            param_dict['resize_calc_ratio'],
          )
        )
      )

    cv2.imshow('Photo', photo_rgb)

    k = cv2.waitKey(0) & 0xFF
    if k == KEY_ESCAPE:
      break
    if k == KEY_ENTER:
      break
    if k == KEY_UP:
      if box_idx > 0:
        box_idx -= 1
    if k == KEY_DOWN:
      if box_idx < len(box_list) - 1:
        box_idx += 1
    if k == KEY_LEFT:
      photo_rgb = cv2.rotate(photo_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if k == KEY_RIGHT:
      photo_rgb = cv2.rotate(photo_rgb, cv2.ROTATE_90_CLOCKWISE)

  cv2.destroyWindow('Photo')


def scale_box(box_tup, scale):
  return [(int(v[0] / scale), int(v[1] / scale)) for v in box_tup]


def debug_show_img(rgb):
  small_rgb = cv2.resize(
    rgb, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
  )
  cv2.imshow('debug', small_rgb)
  cv2.moveWindow('debug', 2000, 0)
  k = cv2.waitKey(0) & 0xFF
  if k == KEY_ESCAPE:
    sys.exit()
  cv2.destroyWindow('debug')


def invert_mask(mask_gray):
  ret, ret_img = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY_INV)
  mask_gray[:] = ret_img


def calc_auto_background_mask(input_rgb, visualize_rgb):
  (x, y, w, h), tolerance, background_mask = calc_best_background_mask(
    input_rgb, visualize_rgb, 1, 5
  )
  visualize_rgb[:] = mask_to_rgb(background_mask)
  visualize_rgb[:, 0, :] = 0
  cv2.rectangle(
    visualize_rgb,
    (x, y),
    (x + w, y + h), color=(255, 255, 0), thickness=3
  )
  return background_mask


def calc_manual_tolerance_mask(
    input_rgb, visualize_rgb, tolerance, sample_step
):
  input_h, input_w = input_rgb.shape[:2]

  combined_mask = unpad_mask(create_blank_mask(input_rgb))

  for x, y in generate_border_coords(input_rgb, sample_step):
    (x, y, w, h), background_mask = calc_flood_area(
      input_rgb, visualize_rgb, x, y, tolerance
    )

    if x == 0 or x == input_w - 1:
      if w == input_w:
        combined_mask = cv2.max(combined_mask, background_mask)

    if y == 0 or y == input_h - 1:
      if h == input_h:
        combined_mask = cv2.max(combined_mask, background_mask)

    # debug_show_img(background_mask)
    # debug_show_img(combined_mask)

  # visualize_rgb[:] = mask_to_rgb(best_background_mask)
  visualize_rgb[:] = mask_to_rgb(combined_mask)
  visualize_rgb[:, 0, :] = 0
  # x, y, w, h = best_box
  cv2.rectangle(
    visualize_rgb,
    (x, y),
    (x + w, y + h), color=(255, 255, 0), thickness=3
  )

  return combined_mask
  # return best_background_mask


def calc_best_background_mask(
    input_rgb, visualize_rgb, tolerance_step, sample_step
):
  """The best background mask is the one that covers the largest area with the
  least tolerance
  """
  best_area = 0
  best_background_mask = None

  input_h, input_w = input_rgb.shape[:2]
  for tolerance in range(0, 255, tolerance_step):
    for x, y in generate_border_coords(input_rgb, sample_step):
      # print('xy {} {} (w h {} {})'.format(x, y, input_w, input_h))
      (x, y, w, h), background_mask = calc_flood_area(
        input_rgb, visualize_rgb, x, y, tolerance
      )

      if w * h > best_area:
        # print('New best')
        best_area = w * h
        best_background_mask = (x, y, w, h), tolerance, background_mask
        # debug_show_img(background_mask)

      # Prioritize getting the largest possible area, but if we already have
      # the whole area, exit early to use the lowest possible tolerance.
      if w == input_w and h == input_h:
        return best_background_mask

  return best_background_mask


def calc_flood_area(input_rgb, visualize_rgb, x, y, tolerance):
  input_gray = cv2.cvtColor(input_rgb, cv2.COLOR_BGR2GRAY)
  fill_mask = create_blank_mask(input_rgb)

  cv2.floodFill(
    input_gray,
    fill_mask,
    (x, y),
    newVal=0, # not used
    loDiff=tolerance,
    upDiff=tolerance,
    flags=(255 << 8) | cv2.FLOODFILL_FIXED_RANGE # | cv2.FLOODFILL_MASK_ONLY
  )

  # cv2.drawMarker(input_gray, (x, y), 255, markerSize=10, thickness=10)
  # debug_show_img(input_gray)

  # IMPORTANT! floodFill() draws a border around the entire image in fill_mask.
  # It's not visible when directly viewing the generated mask as the pixels are
  # of value 1, almost black. The actual touched pixels are value 255.
  ret, fill_mask = cv2.threshold(fill_mask, 128, 255, type=cv2.THRESH_BINARY)

  # debug_show_img(fill_mask)

  workspace_img, contours, hierarchy = cv2.findContours(
    fill_mask,
    cv2.RETR_CCOMP,
    cv2.CHAIN_APPROX_NONE #cv2.CHAIN_APPROX_SIMPLE
  )

  cnt = contours[0]

  # debug_img = input_rgb.copy()
  # cv2.drawContours(
  #   debug_img, [cnt], contourIdx=0, color=(255, 128, 0), thickness=3,
  # )
  # debug_show_img(debug_img)

  x, y, w, h = cv2.boundingRect(cnt)

  # The bounding box is for fill_mask, which is 1 pixel offset from input_rgb
  # in each direction.
  return (x - 1, y - 1, w, h), unpad_mask(fill_mask)


def generate_border_coords(input_rgb, step):
  """Generate coordinates for points along the sides of the image"""
  h, w = input_rgb.shape[:2]
  return ([(x, 0) for x in range(0, w - 1, step)] + [
    (0, y) for y in range(0, h - 1, step)
  ] + [(x, h - 1)
       for x in range(0, w - 1, step)] + [(w - 1, y)
                                          for y in range(0, h - 1, step)])


def find_photos(background_mask, visualize_rgb, param_dict):
  # progress_mask = cv2.cvtColor(out_rgb, cv2.COLOR_BGR2GRAY)

  h, w = background_mask.shape[:2]

  step = 10

  progress_mask = background_mask.copy()

  photo_box_list = []

  for y in range(0, h, step):
    for x in range(0, w, step):
      if progress_mask[y, x] == 255:
        continue

      single_mask = create_blank_mask(background_mask)

      cv2.floodFill(
        progress_mask, single_mask,
        (x, y), newVal=255, loDiff=0, upDiff=0,
        flags=(255 << 8) | cv2.FLOODFILL_FIXED_RANGE
      )

      # IMPORTANT! floodFill() draws a border around the entire image in
      # fill_mask. It's not visible when directly viewing the generated mask as
      # the pixels are of value 1, almost black. The actual touched pixels are
      # value 255.
      ret, single_mask = cv2.threshold(
        single_mask, 128, 255, type=cv2.THRESH_BINARY
      )

      # debug_show_img(single_mask)
      # debug_show_img(progress_mask)

      im2, contours, hierarchy = cv2.findContours(
        single_mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_NONE #cv2.CHAIN_APPROX_SIMPLE
      )

      cnt = contours[-1]
      # x, y, w, h = cv2.boundingRect(cnt)

      min_area_rotated_rect = cv2.minAreaRect(cnt)
      box1 = cv2.boxPoints(min_area_rotated_rect)
      box = numpy.int0(box1)

      cv2.drawContours(
        visualize_rgb,
        [box],
        contourIdx=0,
        color=(255, 128, 0),
        thickness=3,
      )

      # debug_show_img(visualize_rgb)
      photo_box_list.append(box)

  return photo_box_list


def create_blank_rgb(rgb):
  h, w = rgb.shape[:2]
  return numpy.zeros((h, w, 3), numpy.uint8)


def create_blank_mask_unpadded(rgb):
  h, w = rgb.shape[:2]
  return numpy.zeros((h, w), numpy.uint8)


def create_blank_mask(rgb):
  # Size must be 2 pixels more than image in each direction
  h, w = rgb.shape[:2]
  mask = numpy.zeros((h + 2, w + 2), numpy.uint8)
  return mask


def create_pad_mask(mask):
  pmask = create_blank_mask(mask)
  pmask[1:, 1:] = mask
  return pmask


def unpad_mask(mask):
  """Remove the 2x2 pixels of padding that OpenCV requires for masks"""
  return mask[1:-1, 1:-1]


def mask_to_rgb(mask):
  """Remove the 2x2 pixels of padding that OpenCV requires for masks and convert to BGR"""
  return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def find_flood_fill_seed_points(img_rgb):
  """We assume the largest group is the background
  """
  sample_area = 20
  diff_tolerance = 30

  w = sample_area
  h = sample_area

  area_slice_list = [
    (0, w, 0, h),
    (-w, None, 0, h),
    (0, w, -h, None),
    (-w, None, -h, None),
  ]

  area_list = []
  for i, slice_tup in enumerate(area_slice_list):
    x1, y1, x2, y2 = slice_tup
    corner_rgb = img_rgb[x1:y1, x2:y2, :]
    area_list.append((numpy.average(corner_rgb), i, corner_rgb))

  group_dict = {}

  for area_tup in area_list:
    avg, i, rgb = area_tup

    for avg_key in group_dict:
      area_dict = group_dict[avg_key]
      avg_list = area_dict['avg_list']
      this_area_list = area_dict['area_list']

      avg_diff = abs(avg_key - avg)
      if avg_diff < diff_tolerance:
        avg_list.append(avg)
        this_area_list.append(area_tup)
        group_dict[numpy.average(avg_list)] = area_dict
        group_dict.pop(avg_key)
        break
    else:
      group_dict[avg] = {
        'avg_list': [avg],
        'area_list': [area_tup],
      }

  best_key = None
  for avg_key in group_dict:
    if best_key is None or len(group_dict[avg_key]['avg_list']) > len(
        group_dict[best_key]['avg_list']):
      best_key = avg_key

  for area_tup in group_dict[best_key]['area_list']:
    avg, i, rgb = area_tup
    rgb[:] = (0, 255, 255)


def remove_intersecting_smaller_boxes(box_list):
  box_list = [tuple(map(tuple, box_arr)) for box_arr in box_list]

  done_set = set()
  did = True

  while did:
    did = False
    s = set()

    for a, b in itertools.combinations(box_list, 2):
      if a in done_set or b in done_set:
        continue

      box_a = shapely.geometry.LineString((p[1], p[0]) for p in a)
      box_b = shapely.geometry.LineString((p[1], p[0]) for p in b)

      # if box_a.minimum_rotated_rectangle.intersects(box_b.minimum_rotated_rectangle):
      if box_a.convex_hull.intersects(box_b.convex_hull):
        did = True
        if box_a.envelope.area > box_b.envelope.area:
          s.add(a)
          done_set.add(b)
        else:
          done_set.add(a)
          s.add(b)
      else:
        s.add(a)
        s.add(b)

      box_list = list(s)

  box_list = [numpy.array(b) for b in box_list]

  return box_list


def draw_boxes(rgb, box_list, color_tup):
  for box_tup in box_list:
    cv2.drawContours(
      rgb,
      [numpy.array(box_tup)],
      contourIdx=0,
      color=color_tup,
      thickness=4,
    )


def filter_boxes_by_aspect(box_list, min_aspect):
  q = []
  for box_tup in box_list:
    a = shapely.geometry.LineString((box_tup[0], box_tup[1])).length
    b = shapely.geometry.LineString((box_tup[1], box_tup[2])).length
    if a == 0 or b == 0:
      continue
    if a / b < min_aspect or b / a < min_aspect:
      continue
    q.append(box_tup)
  return q


def filter_boxes_by_size(box_list, min_area):
  q = []
  for box_tup in box_list:
    a = shapely.geometry.LineString(box_tup)
    if a.envelope.area >= min_area:
      q.append(box_tup)
  return q


def create_blank_rgb(orig_rgb):
  return numpy.zeros((orig_rgb.shape[0], orig_rgb.shape[1], 3), numpy.uint8)


def create_resized_or_blank_rgb(rgb, w, h):
  if rgb is not None:
    return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LANCZOS4)
  else:
    return numpy.zeros((h, w, 3), numpy.uint8)


def create_tiled_image(img_list, col_count, tot_w, tot_h):
  row_count = math.ceil(len(img_list) / col_count)

  img_w = round(tot_w / col_count)
  img_h = round(tot_h / row_count)

  row_list = []
  for row_i in range(row_count):
    col_list = []
    for col_i in range(col_count):
      col_list.append(
        create_resized_or_blank_rgb((img_list.pop(0)
                                     if img_list else None), img_w, img_h)
      )
    row_list.append(col_list)

  return numpy.vstack((numpy.hstack(r) for r in row_list))


def draw_points(
    dst_rgb, line_pos_list, color_tup=(0, 0, 255), radius=3, thickness=2
):
  for line_tup in line_pos_list:
    x1, y1, x2, y2 = map(int, line_tup)
    cv2.circle(dst_rgb, (x1, y1), radius, color_tup, thickness)


def calc_blur(rgb, param_dict):
  blurred_rgb = cv2.GaussianBlur(
    rgb, (
      param_dict['blur_kernel_size'],
      param_dict['blur_kernel_size'],
    ), param_dict['blur_sigma']
  )
  return blurred_rgb


def extract_and_rotate_photo(full_rgb, box):
  rotated_rect = cv2.minAreaRect(box)
  bound_rect = cv2.boundingRect(box)

  r_rect_angle = rotated_rect[2]
  r_rect_point, r_rect_size = rotated_rect[:2]
  r_rect_w, r_rect_h = int(r_rect_size[0]), int(r_rect_size[1])

  if r_rect_angle <= -45:
    r_rect_angle += 90
    is_rotated = True
  else:
    is_rotated = False

  unrotated_rect = rotated_rect[0], rotated_rect[1], 90
  box11 = cv2.boxPoints(unrotated_rect)
  unrotated_bound_rect = cv2.boundingRect(box11)

  bound_x, bound_y, bound_w, bound_h = bound_rect
  unbound_x, unbound_y, unbound_w, unbound_h = unrotated_bound_rect

  if unbound_w < bound_w:
    unbound_x = bound_x
    unbound_w = bound_w

  if unbound_h < bound_h:
    unbound_y = bound_y
    unbound_h = bound_h

  # print('------------')
  # print(unbound_y, unbound_h)
  #
  # if unbound_y < 0:
  # unbound_rgb = numpy.zeros((unbound_h, unbound_w, 3), numpy.uint8)
  # unbound_rgb[:] = full_rgb[
  #                     unbound_y:unbound_y + unbound_h,
  #                     unbound_x:unbound_x + unbound_w]

  # return numpy.zeros((h, w, 3), numpy.uint8)
  # create_blank_rgb()

  # Can't handle photos that stick outside the border
  full_h, full_w = full_rgb.shape[:2]
  if unbound_y < 0 or unbound_x < 0 or unbound_w > full_w or unbound_h > full_h:
    return

  # For performance, crop to the bounding box
  unbound_rgb = full_rgb[unbound_y:unbound_y + unbound_h, unbound_x:unbound_x +
                         unbound_w]

  # debug_show_img(unbound_rgb)

  # Rotate

  rot_mat = cv2.getRotationMatrix2D((unbound_w / 2, unbound_h / 2),
                                    r_rect_angle, 1)
  rotated_rgb = cv2.warpAffine(
    unbound_rgb, rot_mat,
    (unbound_w, unbound_h), flags=cv2.INTER_CUBIC
  )

  # debug_show_img(rotated_rgb)

  # Final crop

  if is_rotated:
    r_rect_w, r_rect_h = r_rect_h, r_rect_w

  rh_ = (unbound_h - r_rect_h) // 2
  rw_ = (unbound_w - r_rect_w) // 2

  crop_rgb = rotated_rgb[rh_:rh_ + r_rect_h, rw_:rw_ + r_rect_w]

  # print('is_rotated={}'.format(is_rotated))
  # print('unbound_x={}, unbound_y={}, unbound_w={}, unbound_h={}'.format(unbound_x, unbound_y, unbound_w, unbound_h))
  # print('r_rect_w={} r_rect_h={}'.format(r_rect_w, r_rect_h))
  # print('rw_={} rh_={}'.format(rw_, rh_))

  h, w = crop_rgb.shape[:2]

  if h > w:
    crop_rgb = cv2.rotate(crop_rgb, cv2.ROTATE_90_CLOCKWISE)

  # debug_show_img(crop_rgb)

  return crop_rgb


def save_params(param_dict):
  json_str = json.dumps(param_dict, indent=' ', sort_keys=True)
  logging.debug(json_str)
  with open(SETTINGS_PATH, 'w') as f:
    f.write(json_str)
  logging.debug('Saved: {}'.format(SETTINGS_PATH))


def load_params():
  with open(SETTINGS_PATH) as f:
    return json.load(f)


def flat_coords(line_geo):
  """Return (x1, y1, x2, y2) given a shapely.geometry.LineString"""
  c = line_geo.coords
  return c[0][0], c[0][1], c[1][0], c[1][1]


def get_line_geo_by_tup(line_tup):
  """Return a LineString given a flat coordinate tuple"""
  x1, y1, x2, y2 = line_tup
  return shapely.geometry.LineString(((x1, y1), (x2, y2)))


def get_point_tuples_from_flat(flat_tup):
  """(x1, y1, x2, y2, x3, y3) -> ((x1, y1), (x2, y2), (x3, y3)
  """
  return list((flat_tup[i:i + 2][0], flat_tup[i:i + 2][1])
              for i in range(0, len(flat_tup), 2))


def get_flat_from_tuple_of_points(point_tup):
  """((x1, y1), (x2, y2), (x3, y3) -> (x1, y1, x2, y2, x3, y3)
  """
  a = []
  for p in point_tup:
    a.append(p[0])
    a.append(p[1])
  return a


if __name__ == '__main__':
  sys.exit(main())

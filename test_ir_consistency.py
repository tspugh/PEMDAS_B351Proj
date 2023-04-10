import math
import os
import pickle
import sys

from numpy import linspace

import jcamp
from os import path

directory = "new_spectra_all"

def pickle_me(filename):

	p1 = []
	p2 = []
	p3 = []

	for subdir, files, folders in os.walk(directory):

		if subdir.count("/")==2:
			count = 0
			ir_fil = ""
			uv_fil = ""
			for fil in os.listdir(subdir):
				if ".jdx" in fil or ".csv" in fil:
					if "_IR_" in fil:
						ir_fil = os.path.join(subdir,fil)
					if "_UV_" in fil:
						uv_fil = os.path.join(subdir,fil)
					if "_MS_" in fil:
						ms_fil = os.path.join(subdir,fil)
					count += 1
			if count == 5:
				try:
					p = jcamp.jcamp_readfile(ir_fil)
					qq = jcamp.jcamp_readfile(uv_fil)
					qqq = jcamp.jcamp_readfile(ms_fil)
					m = p["x"][0]
					n = qq["x"][0]
					o = qqq["x"][0]
					p1.append(ir_fil)
					p2.append(uv_fil)
					p3.append(ms_fil)
				except Exception:
					pass
	ally = [p1, p2, p3]

	print(ally)
	file_to_write = open(filename, "wb")
	pickle.dump(ally, file_to_write, protocol=pickle.HIGHEST_PROTOCOL)
	file_to_write.close()
	return ally

def find_info(arraythang):

	min = -math.inf
	ave_start = 0
	max = math.inf
	ave_end = 0
	max_length = math.inf
	c_count = 0

	for ir_fil in arraythang:
		try:
			p = jcamp.jcamp_readfile(ir_fil)
			m = p["x"][0]
			ma = p["x"][-1]
			if m > min:
				min = m
			if ma < max:
				max = ma
			if len(p["x"]) < max_length:
				max_length = len(p["x"])

			c_count += 1

			if ave_start == 0:
				ave_start = m
			else:
				ave_start = (ave_start * (c_count - 1) + m) / c_count

			if ave_end == 0:
				ave_end = ma
			else:
				ave_end = (ave_end * (c_count - 1) + ma) / c_count


		except Exception:
			print("me cry")

	print(f"min: {min}, max: {max}, len: {max_length}")
	print(f"average start: {ave_start}, average end: {ave_end}")

def find_inclusive(ir_dirs,start, end, step):
	init_len = start
	for x in linspace(init_len, end, step):
		counter = 0
		for i in ir_dirs:
			p = jcamp.jcamp_readfile(i)
			if len(p["x"]) > x:
				counter += 1
		print(f"for min len {x}: {counter}")

def triple_find_inclusive(dirs, count):

	ms = linspace(10, 120, count)
	ir = linspace(800, 5000, count)
	uv = linspace(20, 500, count)

	for i in range(count):
		counter = 0
		for j in range(len(dirs[0])):
			p = jcamp.jcamp_readfile(dirs[0][j])
			q = jcamp.jcamp_readfile(dirs[1][j])
			r = jcamp.jcamp_readfile(dirs[2][j])
			if len(p["x"]) > ir[i]  and len(q["x"]) > \
			                                       uv[i] and len(
				r["x"]) > ms[i]:
				counter += 1
		print(f"for min len {uv[i]},{ir[i]},{ms[i]}: {counter}")


def get_from_pickle(filename):

	file_to_read = open(filename, "rb")
	return pickle.load(file_to_read)


if __name__ == "__main__":
	sys.stdout = open('result_of_spectra.txt', 'wt')

	my_test_file = "ir_file_to_test_what_to_do.txt"
	dirs = get_from_pickle(my_test_file)

	"""
	print("Mass spectra")
	find_info(dirs[2])
	print()
	find_inclusive(dirs[2], 10, 130, 50)

	print("\nUV spectra")
	find_info(dirs[1])
	print()
	find_inclusive(dirs[1], 70, 300, 50)

	print("\nIR spectra")
	find_info(dirs[0])
	print()
	find_inclusive(dirs[0], 820, 5000, 50)
	"""

	triple_find_inclusive(dirs, 100)


	pass







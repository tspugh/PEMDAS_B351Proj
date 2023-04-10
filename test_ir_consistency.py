import itertools
import math
import os
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np

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
	for x in np.linspace(init_len, end, step):
		counter = 0
		for i in ir_dirs:
			p = jcamp.jcamp_readfile(i)
			if len(p["x"]) > x:
				counter += 1
		print(f"for min len {x}: {counter}")

def triple_find_inclusive(dirs, count):

	ms = np.linspace(10, 120, count)
	ir = np.linspace(800, 5000, count)
	uv = np.linspace(20, 500, count)

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

def interpolate(ir_file, uv_file, ms_file):

	ir_noise = 0.01
	ms_noise = 50


	ir = jcamp.jcamp_readfile(ir_file)
	uv = jcamp.jcamp_readfile(uv_file)
	ms = jcamp.jcamp_readfile(ms_file)

	if ir["xunits"] == "MICROMETERS":
		ir["xunits"] = "1/CM"
		ir["x"] *= 1000

	start_index = 0
	maximum = len(ir["y"])
	while start_index < maximum:

		if ir["y"][start_index] < ir_noise:
			if start_index == 0 or start_index == maximum-1 or \
					(ir["y"][start_index-1] < ir_noise and ir["y"][
						start_index+1] < ir_noise):
				end_index = start_index
				while end_index < maximum-1 and ir["y"][end_index+1] < \
						ir_noise:
					end_index += 1
				ir["x"] = [ ir["x"][j] for j in itertools.chain(range(0,
				                                          start_index),
				                                  range(end_index,
				                                        maximum))]
				ir["y"] = [ir["y"][j] for j in itertools.chain(range(0,
				                                          start_index),
				                                  range(end_index,
				                                        maximum))]
		start_index += 1
		maximum = len(ir["y"])
	ir["y"] = ir["y"][:maximum]
	ir["x"] = ir["x"][:maximum]

	start_index = 0
	maximum = len(ms["y"])
	while start_index < maximum:
		maximum = len(ms["y"])
		if ms["y"][start_index] < ms_noise:
			if start_index == 0 or start_index == maximum - 1 or \
					(ms["y"][start_index - 1] < ms_noise and ms["y"][
						start_index + 1] < ms_noise):
				end_index = start_index
				while end_index < maximum-1 and ms["y"][end_index + 1]\
						< \
						ms_noise:
					end_index += 1
				ms["x"] = [ms["x"][j] for j in itertools.chain(range(0,
				                                          start_index),
				                                  range(end_index,
				                                        maximum))]
				ms["y"] = [ms["y"][j] for j in itertools.chain(range(0,
				                                          start_index),
				                                  range(end_index,
				                                        maximum))]
		start_index+=1
		maximum = len(ms["y"])
	ms["x"] = ms["x"][:maximum]
	ms["y"] = ms["y"][:maximum]

	return ir, uv, ms



def refine_and_load(dirs, ir_cut, uv_cut, ms_cut):

	ir = []
	uv = []
	ms = []


	for j in range(len(dirs[0])):
		ir_j, uv_j, ms_j = interpolate(dirs[0][j],dirs[1][j],dirs[2][j])
		if len(ir_j["x"]) < ir_cut and len(uv_j["x"]) < \
				uv_cut and len(
			ms_j["x"]) < ms_cut:
			ir.append(dirs[0][j])
			uv.append(dirs[1][j])
			ms.append(dirs[2][j])
	new_dirs = [ir, uv, ms]
	return new_dirs

def fourier_baby(dirs):
	#
	for j in range(len(dirs[0])):
		ir = jcamp.jcamp_readfile(dirs[0][j])
		uv = jcamp.jcamp_readfile(dirs[1][j])
		ms = jcamp.jcamp_readfile(dirs[2][j])

		f_ir = np.fft.ifft(ir["y"])
		n = np.arange(len(f_ir))
		freq = n/len(ir["x"]) * len(f_ir)


		plt.stem(freq, np.abs(f_ir), "b")
		plt.show()

def show_plots(dirs):
	for j in range(len(dirs[0])):
		ir, uv, ms = interpolate(dirs[0][j],dirs[1][j],dirs[2][j])

		if ir["xunits"] == "MICROMETERS":
			ir["xunits"] = "1/CM"
			ir["x"] *= 1000

		plt.figure(figsize=(12, 4))
		plt.subplot(131)
		plt.plot(ir["x"],ir["y"],'r')
		plt.xlabel(ir["xunits"])
		plt.ylabel(ir["yunits"])
		plt.subplot(132)
		plt.plot(uv["x"], uv["y"], 'b')
		plt.xlabel(uv["xunits"])
		plt.ylabel(uv["yunits"])
		plt.subplot(133)
		plt.plot(ms["x"], ms["y"], 'g')
		plt.xlabel(ms["xunits"])
		plt.ylabel(ms["yunits"])
		plt.show()

def get_from_pickle(filename):

	file_to_read = open(filename, "rb")
	return pickle.load(file_to_read)

def save_and_cut(dirs):

	ir_cut = 200
	uv_cut = 1500
	ms_cut = 100
	new_file = f"refinement_{ir_cut}-{uv_cut}-{ms_cut}.txt"
	new_dirs = refine_and_load(dirs, ir_cut, uv_cut, ms_cut)

	file_to_write = open(new_file, "wb")
	pickle.dump(new_dirs, file_to_write,
	            protocol=pickle.HIGHEST_PROTOCOL)
	file_to_write.close()


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

	save_and_cut(dirs)



	pass







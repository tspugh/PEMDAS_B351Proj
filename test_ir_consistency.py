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

def find_info(arraythang, min_length):

	min = math.inf
	ave_start = 0
	max = -math.inf
	ave_end = 0
	max_length = math.inf
	c_count = 0



	for ir_fil in arraythang:
		try:
			p = jcamp.jcamp_readfile(ir_fil)
			if p["x"][0] > p["x"][1]:
				p["x"] = p["x"][-1:0:-1]
				p["y"] = p["y"][-1:0:-1]

			if p["xunits"] == "MICROMETERS":
				p["xunits"] = "1/CM"
				p["x"] *= 1000

			if len(p["x"])> min_length:

				m = p["x"][0]
				ma = p["x"][-1]
				if m < min:
					min = m
				if ma > max:
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

	print(f"min: {min}, max: {max}, len: {c_count}")
	print(f"average start: {ave_start}, average end: {ave_end}")

def try_mins(ir_spectra):
	for x in np.linspace(0, 30000, 1000):
		counter = 0
		for i in ir_spectra:
			p = jcamp.jcamp_readfile(i)
			if p["x"][0] > 200 + x*200/30000 and p["x"][-1] < 38000 - x:
				counter += 1
		print(f"for {200 + x*200/30000} to {200 + x*200/30000}: {counter}")

def find_inclusive(ir_dirs,start, end, step):
	init_len = start
	for x in np.linspace(init_len, end, step):
		counter = 0
		for i in ir_dirs:
			p = jcamp.jcamp_readfile(i)
			if len(p["x"]) > x:
				counter += 1
		print(f"for min len {x}: {counter}")

def find_inclusive_new(ir_dirs):
	for x in range(800, 5000, 300):
		print(f"for min len {x}:")
		find_info(ir_dirs, x)



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



	start_index = 0
	maximum = len(ir["y"])
	"""
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
	"""

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


		if(len(ir["wavenumbers"]) == len(ir["y"]) and len(uv[
			                                                  "wavenumbers"]) == len(uv["y"])):
			plt.figure(figsize=(12, 4))
			plt.subplot(121)
			plt.plot(ir["wavenumbers"],ir["y"],'r')
			plt.xlabel(ir["xunits"])
			plt.ylabel(ir["yunits"])
			plt.subplot(122)
			plt.plot(uv["wavenumbers"], uv["y"], 'b')
			plt.xlabel(uv["xunits"])
			plt.ylabel(uv["yunits"])

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


def test_mins_ir(dirs):

	MIN_IR = 600
	MAX_IR = 3500


	MIN_UV = 300
	MAX_UV = 780

	max_uv_len = 0

	count_ir = 0
	count_uv = 0

	tries = 20
	counts = np.zeros(tries)
	max_ir_len = np.zeros(tries)

	for j in range(len(dirs[0])):
		ir = jcamp.jcamp_readfile(dirs[0][j])
		uv = jcamp.jcamp_readfile(dirs[1][j])

		if not ir["x"][0]<ir["x"][1]:
			ir["x"] = ir["x"][::-1]
			ir["y"] = ir["y"][::-1]

		if not uv["x"][0]<uv["x"][1]:
			uv["x"] = uv["x"][::-1]
			uv["y"] = uv["y"][::-1]

		index = 0
		for lower in np.linspace(3700, 4000, tries):
			if(ir['xunits'].lower() != "micrometers" and ir["x"][0]<=MIN_IR
					and
					ir["x"][-1] >= lower):
				counts[index] += 1
				if len(ir["x"])>max_ir_len[index]:
					max_ir_len[index] = len(ir["x"])
			index += 1

		"""
		if (uv['xunits'].lower() != "micrometers" and uv["x"][
			0] <= MIN_UV
				and
				uv["x"][-1] >= MAX_UV):
			count_uv += 1
			if len(uv["x"]) > max_uv_len:
				max_uv_len = len(uv["x"])
				"""

	for x in range(tries):
		print(f"bound of {3700+(4000-3700)/tries*x}:\n")
		print(f"ir: {counts[x]}, max length: {max_ir_len[x]}")

def test_mins_uv(dirs):
	# 225 - 250

	MIN_UV = 225
	MAX_UV = 250

	max_uv_len = 0

	count_ir = 0
	count_uv = 0

	tries = 20
	counts = np.zeros(tries)
	max_uv_len = np.zeros(tries)

	for j in range(len(dirs[0])):
		uv = jcamp.jcamp_readfile(dirs[1][j])

		if not uv["x"][0]<uv["x"][1]:
			uv["x"] = uv["x"][::-1]
			uv["y"] = uv["y"][::-1]

		index = 0
		for lower in np.linspace(225, 350, tries):
			if(uv["x"][0]<=MIN_UV
					and
					uv["x"][-1] >= lower):
				counts[index] += 1
				if len(uv["x"])>max_uv_len[index]:
					max_uv_len[index] = len(uv["x"])
			index += 1

		"""
		if (uv['xunits'].lower() != "micrometers" and uv["x"][
			0] <= MIN_UV
				and
				uv["x"][-1] >= MAX_UV):
			count_uv += 1
			if len(uv["x"]) > max_uv_len:
				max_uv_len = len(uv["x"])
				"""

	for x in range(tries):
		print(f"bound of {225+(350-225)/tries*x}:\n")
		print(f"uv: {counts[x]}, max length: {max_uv_len[x]}")

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

	test_mins_uv(dirs)



	pass







import os, sys
from shutil import copyfile
from pubchem_reader import meets_criteria


directory = "../spectra/1-50001"
NMR_1 = "../NMR/1HNMR"
NMR_2 = "../NMR/13CNMR"

def process_folders():
	for subdir, folders, files in os.walk(directory):
		for f in folders:
			dir = os.path.join(subdir, f)
			if dir.count("/")==3:
				values = meets_criteria(f)
				print(values)
				if values is not None and len(values)==2 and values[0]:
					smiles = values[1]["CanonicalSMILES"]
					ms = None
					uv = None
					ir = None
					hnmr = None
					cnmr = None
					for file in os.listdir(dir) :
						q = os.path.join(dir,file)
						if os.path.isfile(q):
							if ms is None and q.count("_MS_")==1:
								ms=q
							elif uv is None and q.count("_IR_")==1:
								uv=q
							elif ir is None and q.count("_UV_")==1:
								ir=q
							elif not (ir is None or uv is None or ms is
							          None):
								break
								"""
					if not (ir is None or uv is None or ms is None):
						for file in os.listdir(NMR_1):
							if hnmr is None and f in file:
								hnmr=os.path.join(NMR_1,file)
								break
						if hnmr is not None:
							for file in os.listdir(NMR_2):
								if cnmr is None and f in file:
									cnmr = os.path.join(NMR_2, file)
									break
									"""
					if not (ms is None or uv is None or ir is None):
						#or cnmr is None or hnmr is None):

						pth = f"../new_spectra_2/{f}"
						os.makedirs(pth)
						_, end = os.path.split(ms)
						copyfile(ms, os.path.join(pth,end))
						_, end = os.path.split(uv)
						copyfile(uv, os.path.join(pth, end))
						_, end = os.path.split(ir)
						copyfile(ir, os.path.join(pth, end))
						"""
						_, end = os.path.split(cnmr)
						copyfile(cnmr, os.path.join(pth, "13CNMR_"+end))
						_, end = os.path.split(hnmr)
						copyfile(hnmr, os.path.join(pth, "1HNMR_"+end))
						"""
						q = open(os.path.join(pth,
						                 "classification_info.txt"),"w")
						q.write(f+"\n"+smiles+"\n"+"Negative")
						q.close()




if __name__ == "__main__":
	process_folders()
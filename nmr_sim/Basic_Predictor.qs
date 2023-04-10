
// Clear pages from the current document and re-init with a fresh page.
clear_document = function(doc) {
	'use strict';
	var count = doc.pageCount();
	for ( var i = 0 ; i < count ; i++ ) {
		doc.deletePages(0);	
	}
	doc.addPage();
};	

// Given a molecule name and spectrum type ("1H" or "13C"),
//   return the filepath to the output file for the spectrum.
get_directory = function (name, type) {
	'use strict';
	return Dir.home() + "/" + PROJECT_DIR + "/" + name;
};
get_filepath = function (name, type) {
	'use strict';
	return get_directory(name,type) + "/" + type + ".csv";
};	

save_spectrum_to_file = function (spec, filepath) {
	'use strict';
	if ( spec == undefined || !spec.isValid() ) { return; }
	var file = new File(filepath);
	var mapObj = {},
	    aDecimals = 6, rounder,
	    aFormat = "{ppm},{real}",
	    hz, dHz, pt, dPt, endPt, ppm, dPpm, strm;	    
	
	if (file.open(File.WriteOnly)) {
		strm = new TextStream(file);	
		
		hz = spec.hz() + spec.scaleWidth();
		dHz = -spec.scaleWidth() / spec.count();		
		pt = spec.count()-1;
		dPt = -1;
		endPt = -1;
		
		rounder = Math.pow(10,aDecimals);
		ppm = Math.round(hz / spec.frequency() * rounder) / rounder;
		dPpm = Math.round(dHz / spec.frequency() * rounder) / rounder;		
		
		while ( pt !== endPt ) {		
			//mapObj.hz = hz.toFixed(aDecimals);
			mapObj.ppm = ppm.toFixed(aDecimals);
			//mapObj.pts = pt;
			mapObj.real = spec.real(pt).toFixed(aDecimals);				
			//mapObj.imag = spec.imag(pt).toFixed(aDecimals);
			// Compression: Don't write 0s. This hugely reduces file size and running time.
			if ( mapObj.real != 0 ) {
				strm.writeln(aFormat.formatMap(mapObj));					
			}
			//hz += dHz;
			ppm += dPpm;
			pt += dPt;
		}
		file.close();
		print("Saved to " + filepath);
	} else { print("Problem writing to " + filepath); }		
};

save_dummy_spectrum = function(filepath) {
	'use strict';
	var file = new File(filepath);	
//	var strm = new TextStream(file);
//	strm.writeln("");
	if ( file.open(File.WriteOnly)) {
		file.close();
		print("Saved empty spectrum to " + filepath);
	} else { print("Problem writing empty spectrum to " + filepath); }
};

molecule_contains = function(mol, element) {
	'use strict';
	var arr = mol.atoms();	
	for ( var i = 0 ; i < arr.length ; i++ ) {
		if ( arr[i].elementSymbol == element ) { return true; }
	}
	return false;
};

make_predictions = function (mol) {
	'use strict';
	if ( mol == undefined || !mol.isValid() ) { return; }
	//print(mol.generateSMILES());
	
	var spec, spec_1H, spec_13C;
	
	// If the output files already exist, skip.
	var shouldMake1H = true, shouldMake13C = true;
	if ( File.exists(get_filepath(mol.molName,"1H")) ) {
		shouldMake1H = false;
	}
	if ( File.exists(get_filepath(mol.molName,"13C")) ) {
		shouldMake13C = false;
	}
	// Optimization: Immediately end if both files exist...
	if ( !shouldMake1H && !shouldMake13C ) { return; }
	
	// Also check to see if we'll even get spectra.
	// Check to see if the molecule has hydrogens...
	mol.addExplicitHydrogens();
	//shouldMake1H = shouldMake1H && molecule_contains(mol,"H");
	//shouldMake13C = shouldMake13C && molecule_contains(mol,"C");	
	
	if ( shouldMake1H && !molecule_contains(mol,"H") ) {
		shouldMake1H = false;
		save_dummy_spectrum(get_filepath(mol.molName,"1H"));	  
	}
	if ( shouldMake13C && !molecule_contains(mol,"C") ) {
		shouldMake13C = false;
		save_dummy_spectrum(get_filepath(mol.molName,"13C"));	  
	}
	
	// Optimization: Immediately end if both files exist...	
	if ( !shouldMake1H && !shouldMake13C ) { return; }
	
	// Predict 1H and 13C NMR spectra. This sticks them on the current document.
	if ( shouldMake1H ) {
		print("Predicting 1H for " + mol.molName);
		Application.NMRPredictor.predict(mol, "1H");
	}
	if ( shouldMake13C ) {
		print("Predicting 13C for " + mol.molName);
		Application.NMRPredictor.predict(mol, "13C");
	}
	
	// Find the spectra we just made...
	var document = Application.mainWindow.activeDocument;
	for ( var i = 0 ; i < document.itemCount() ; i++ ) {
		spec = new NMRSpectrum(document.item(i));	
		if ( !spec.isValid() ) { continue; } // Oops this isn't a spectrum.
		// If this is an NMR spectrum assume it's one we just made.
		
		if ( spec.title == "Predicted 1H NMR Spectrum" ) { spec_1H = spec; }
		else if ( spec.title == "Predicted 13C NMR Spectrum" ) { spec_13C = spec; }
	}
	//Custom1DCsvConverter.formattedExport(
	//	Application.mainWindow.activeDocument.pages(),Dir.home() + "/Custom1DCsv.txt",
	//	"{ppm}{tab}{real}{tab}{imag}",6,false);	
	
	// Save to file.
	//var 1Hout = MOL_1H_DIR + name + ".csv";
	//var 13Cout = MOL_13C_DIR + name + ".csv";
	if ( spec_1H != undefined ) { save_spectrum_to_file(spec_1H,get_filepath(mol.molName,"1H"));	 }
	if ( spec_13C != undefined ) { save_spectrum_to_file(spec_13C,get_filepath(mol.molName,"13C")); }
};

// Make a prediction for some .mol file and save those predictions to file.
predict_for_file = function (filepath) {
	'use strict';
	var name;
	// Remove the filepath and only keep the filename.
	var filearr;
	filearr = filepath.split("/");
	name = filearr[filearr.length - 1 ];
	filearr = name.split("\\");
	name = filearr[filearr.length - 1];
	// Remove the file extension.
	name = name.replace(".mol","");
	
	// Clear the current document,
	var doc = Application.mainWindow.activeDocument;
	clear_document(doc);
	// Try to open the molecule file that we were passed.
	print("Opening " + filepath);
	var status = serialization.open(filepath, "molfile");
	print("Success = " + status);
	// This has opened this molecule "page" on the new document.
	//   Try to find the molecule object on the page.
	for ( var i = 0; i < doc.itemCount() ; i++ ) {
		// Make a copy of the object and see if it works as a molecule.		
		var mol = new Molecule(doc.item(i));
		if ( !mol.isValid() ) { continue; } // This isn't actually a molecule, try other objects.
		// Note: This method works if we have multiple molecules in the document,
		//   except we would need to name them different things.
		mol.molName = name; // Store the name in a molecule field.
		// Make the predictions and store them to file.
		make_predictions(mol);
	}		
};

// Make a prediction for a SMILES string of some name and save those predictions to file.
predict_for_smiles = function(smiles, name) {
	'use strict';
	// Clear the current document.
	var doc = Application.mainWindow.activeDocument;
	clear_document(doc);
	// Add the SMILES molecule.
	molecule.importSMILES(smiles);
	for ( var i = 0 ; i < doc.itemCount() ; i++ ) {
		// Look for the molecule in the document. Is this item the molecule we just made?
		var mol = new Molecule(doc.item(i));
		if ( !mol.isValid() ) { continue; } // Not a molecule. Try again.
		mol.molName = name;
		make_predictions(mol);
	}
};

// Iterate through .mol files in a directory and make predictions for each.
batch_prediction_mol = function(directory) {
	'use strict';
	print("Batch predicting in " + directory);
	// Iterate through .mol files in this directory and generate spectra for them.
	var dir = Dir(directory); // This object lets us traverse the directory.
	var entries = dir.entryList("*.mol",Dir.Files); // Array of valid file names.	
	for ( var i = 0 ; i < entries.length ; i++ ) {
		//print("Predicting for " + entries[i]);
		predict_for_file(directory + "/" + entries[i]);
	}
	print("Done!");
};

// Iterate through SMILES~Name pairs in a text file and generate predictions for each.
batch_prediction_smiles = function(filepath) {
	'use strict';
	print("Batch predicting using " + filepath);
	var file = new File(filepath);
	if ( file == undefined ) { print("Problem opening " + filepath); return; }
	if ( !file.open(File.ReadOnly) ) { print("Problem opening " + filepath); return; }
	var strm = new TextStream(file);
	var line = strm.readLine();
	var arr;
	do {
		arr = line.split("~"); // This is the delimiter used in the text file to separate SMILES and name.
		// Forward check to see if spectra already exist before doing anything.
		// Even though this check is done again later, this is SUPER fast for running through spectra that we've already simulated
		if ( Dir(get_directory(arr[1])).exists ) {
			if ( !File.exists(get_filepath(arr[1],"1H")) || !File.exists(get_filepath(arr[1],"13C")) ) {
				predict_for_smiles(arr[0],arr[1]);			
			}
		}
		line = strm.readLine();
	} while ( line != "" );
	file.close();
	print("Done!");
};	

batch_prediction_smiles(Dir.home() + "/" + SMILES_FILE);

//predict_for_smiles("cc", "test");

//batch_prediction(MOL_IN_DIR);

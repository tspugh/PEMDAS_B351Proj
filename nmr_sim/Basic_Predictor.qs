
save_spectrum_to_file = function (spec, filepath) {
	'use strict';
	if ( spec == undefined || !spec.isValid() ) { return; }
	var file = new File(filepath);
	var mapObj = {},
	    aDecimals = 6,
	    aFormat = "{ppm},{real}",
	    hz, dHz, pt, dPt, endPt, ppm, dPpm, ticker, strm;
	
	if (file.open(File.WriteOnly)) {
		strm = new TextStream(file);	
		
		hz = spec.hz() + spec.scaleWidth();
		dHz = -spec.scaleWidth() / spec.count();
		pt = 0;
		dPt = 1;
		endPt = spec.count();
		
		ppm = hz / spec.frequency();
		dPpm = dHz / spec.frequency();			
		while ( pt !== endPt ) {
			ticker = 0;
			mapObj.hz = hz.toFixed(aDecimals);
			mapObj.ppm = ppm.toFixed(aDecimals);
			mapObj.pts = pt;
			mapObj.real = spec.real(pt).toFixed(aDecimals);				
			//mapObj.imag = spec.imag(pt).toFixed(aDecimals);
			// Compression: Don't write 0s. This hugely reduces file size and running time.
			if ( mapObj.real != 0 ) {
				strm.writeln(aFormat.formatMap(mapObj));					
			}
			hz += dHz;
			ppm += dPpm;
			pt += dPt;
		}
		file.close();
		print("Saved to " + filepath);
	} else { print("Problem writing to " + filepath); }		
};

make_predictions = function (mol) {
	'use strict';
	if ( mol == undefined || !mol.isValid() ) { return; }
	//print(mol.generateSMILES());
	
	var spec, spec_1H, spec_13C;
	
	// Predict 1H and 13C NMR spectra. This sticks them on the current document.
	Application.NMRPredictor.predict(mol, "1H");
	Application.NMRPredictor.predict(mol, "13C");
	
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
	if ( spec_1H != undefined ) { save_spectrum_to_file(spec_1H,MOL_1H_DIR + mol.molName + ".csv");	 }
	if ( spec_13C != undefined ) { save_spectrum_to_file(spec_13C,MOL_13C_DIR + mol.molName + ".csv"); }
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
	
	// Make a new blank document (it's a tab in the current window).
	var aDocument = new Document();
	Application.mainWindow.addDocument(aDocument);
	// Try to open the molecule file that we were passed.
	print("Opening " + filepath);
	var status = serialization.open(filepath, "molfile");
	print("Success = " + status);
	// This has opened this molecule "page" on the new document.
	//   Try to find the molecule object on the page.
	for ( var i = 0; i < aDocument.itemCount() ; i++ ) {
		// Make a copy of the object and see if it works as a molecule.		
		var mol = new Molecule(aDocument.item(i));
		if ( !mol.isValid() ) { continue; } // This isn't actually a molecule, try other objects.
		// Note: This method works if we have multiple molecules in the document,
		//   except we would need to name them different things.
		mol.molName = name; // Store the name in a molecule field.
		// Make the predictions and store them to file.
		make_predictions(mol);
	}
	aDocument.destroy(); // Delete that document so we don't accumulate tabs.
};

// Make a prediction for a SMILES string of some name and save those predictions to file.
predict_for_smiles = function(smiles, name) {
	'use strict';
	name = name.replace(" ","_"); // Avoid spaces in filenames so make the name not have spaces.
	// Make a new blank document.
	var aDocument = new Document();
	Application.mainWindow.addDocument(aDocument);
	// Add the SMILES molecule.
	molecule.importSMILES(smiles);
	for ( var i = 0 ; i < aDocument.itemCount() ; i++ ) {
		// Look for the molecule in the document. Is this item the molecule we just made?
		var mol = new Molecule(aDocument.item(i));
		if ( !mol.isValid() ) { continue; } // Not a molecule. Try again.
		mol.molName = name;
		make_predictions(mol);
	}
	aDocument.destroy(); // Delete that document so we don't accumulate tabs.
};

// Iterate through .mol files in a directory and make predictions for each.
batch_prediction_mol = function(directory) {
	'use strict';
	print("Batch predicting in " + directory);
	// Iterate through .mol files in this directory and generate spectra for them.
	var dir = Dir(directory); // This object lets us traverse the directory.
	var entries = dir.entryList("*.mol",Dir.Files); // Array of valid file names.	
	for ( var i = 0 ; i < entries.length ; i++ ) {
		print("Predicting for " + entries[i]);
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
		predict_for_smiles(arr[0],arr[1]);
		print("Predicting for " + arr[1]);
		line = strm.readLine();
	} while ( line != "" );
	file.close();
	print("Done!");
};

//batch_prediction_smiles(SMILES_FILE);

//batch_prediction(MOL_IN_DIR);
//molecule.importSMILES("cc");

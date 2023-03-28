
save_spectrum_to_file = function (spec, filepath) {
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
		var ticker = 0; // Skip every 5 points to reduce file size...
		while ( pt !== endPt ) {
			//if ( ticker < 5 ) {	 ticker++; }
			//else {
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
			//}
			hz += dHz;
			ppm += dPpm;
			pt += dPt;					
		}
		file.close();
		print("Saved to " + filepath);
	} else { print("Problem writing to " + filepath); }		
};

make_predictions = function (mol) {	
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

predict_for_file = function (filepath) {
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
		var molecule = new Molecule(aDocument.item(i));
		if ( !molecule.isValid() ) { continue; } // This isn't actually a molecule, try other objects.
		// Note: This method works if we have multiple molecules in the document,
		//   except we would need to name them different things.
		molecule.molName = name; // Store the name in a molecule field.
		// Make the predictions and store them to file.
		make_predictions(molecule);
	}
	aDocument.destroy(); // Delete that document so we don't accumulate tabs.
};

batch_prediction = function(directory) {
	print("Batch predicting in " + directory);
	// Iterate through .mol files in this directory and generate spectra for them.
	var dir = Dir(directory); // This object lets us traverse the directory.
	var entries = dir.entryList("*.mol",Dir.Files); // Array of valid file names.	
	for ( var i = 0 ; i < entries.length ; i++ ) {
		print(i + ": " + entries[i]);	
		predict_for_file(directory + "/" + entries[i]);
	}
	print("Done!");
};

//batch_prediction(MOL_IN_DIR);


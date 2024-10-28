window.addEventListener('load', function() {
    // Hide the loader
    document.getElementById('loader').style.display = 'none';
    
    // Show the main content
    document.getElementById('content').style.display = 'block';
});

// $(document).ready(function() {
    // Handle input formatting to MM-DD-YYYY
    // $('#dob').on('input', function() {
    //     let input = $(this).val().replace(/\D/g, '').substring(0, 8);
    //     let month = input.substring(0, 2);
    //     let day = input.substring(2, 4);
    //     let year = input.substring(4, 8);

    //     if (input.length >= 5) {
    //         $(this).val(`${month}-${day}-${year}`);
    //     } else if (input.length >= 3) {
    //         $(this).val(`${month}-${day}`);
    //     } else {
    //         $(this).val(month); // Ensure it shows only the month if less than 3 characters
    //     }
    // });

    // Validate the date on blur
//     $('#dob').on('blur', function() {
//         let input = $(this).val();
//         let dateParts = input.split('-');

//         if (dateParts.length === 3) {
//             let month = parseInt(dateParts[0], 10);
//             let day = parseInt(dateParts[1], 10);
//             let year = parseInt(dateParts[2], 10);
//             let inputDate = new Date(year, month - 1, day); // month is 0-indexed in JS
//             let today = new Date();

//             // Check if the date is valid
//             if (isNaN(inputDate) || month < 1 || month > 12 || day < 1 || day > 31 || inputDate.getFullYear() !== year) {
//                 alert('Invalid date. Please enter a valid date in the format MM-DD-YYYY.');
//                 $(this).val('');
//             } 
//             // Check if the date is in the future
//             else if (inputDate > today) {
//                 alert('The date cannot be in the future. Please select a valid date of birth.');
//                 $(this).val('');
//             } 
//             // Check if the year is before 1900
//             else if (year < 1900) {
//                 alert('The year must be 1900 or later. Please enter a valid year.');
//                 $(this).val('');
//             }
//         } else {
//             // Incomplete date input
//             alert('Please enter the complete date in MM-DD-YYYY format.');
//             $(this).val('');
//         }
//     });
// });

$(document).ready(function () {
    // Add shadow to navbar on scroll
    $(window).on('scroll', function () {
        var $navbar = $('.navbar-custom');
        if ($(this).scrollTop() > 0) {
            $navbar.addClass('scrolled');
        } else {
            $navbar.removeClass('scrolled');
        }
    });

    // Format phone number as user types
    function formatPhoneNumber(value) {
        var cleaned = ('' + value).replace(/\D/g, ''); // Remove non-digit characters
        var match = cleaned.match(/^(\d{3})(\d{3})?(\d{4})$/);
        if (match) {
            return '(' + match[1] + ') ' + (match[2] ? match[2] + '-' : '') + (match[3] || '');
        }
        return value;
    }

    // Format phone number on input
    $('#phonenumber').on('input', function (e) {
        var formatted = formatPhoneNumber(e.target.value);
        $(this).val(formatted.slice(0, 14)); // Restrict to 14 characters
    });

    // Validate phone number on form submission
    // $('#userForm').on('submit', function (event) {
    //     var $input = $('#phonenumber');
    //     var phonePattern = /^\(\d{3}\) \d{3}-\d{4}$/;
    //     var cleaned = $input.val().replace(/\D/g, '');

    //     if (cleaned.length !== 10) {
    //         event.preventDefault(); // Prevent form submission
    //         $('#phoneError').text('Phone number must be 10 digits.');
    //     } else if (!phonePattern.test($input.val())) {
    //         event.preventDefault(); // Prevent form submission
    //         $('#phoneError').text('Phone number format should be (123) 456-7890.');
    //     } else {
    //         $('#phoneError').text('');
    //     }
    });

    // Function to show/hide bundle field
    function toggleBundleField() {
        if ($('#homeownerYes').is(':checked')) {
            $('#bundle-field').removeClass('d-none');
        } else {
            $('#bundle-field').addClass('d-none');
        }
    }

    // Event listeners for homeowner toggle
    $('input[name="homeowner"]').change(toggleBundleField);


    // // Function to show/hide additional vehicles info
    // function toggleAdditionalVehiclesInfo() {
    //     if ($('#vehiclesYes').is(':checked')) {
    //         $('#additional-vehicles-info').show(); // Show the div
    //     } else {
    //         $('#additional-vehicles-info').hide(); // Hide the div
    //     }
    // }

//     // Initial check to set the visibility based on the default checked radio button
//     toggleAdditionalVehiclesInfo();

//     // Attach event listener to the radio buttons
//     $('input[name="multiple-vehicles"]').change(toggleAdditionalVehiclesInfo);

// });

// $(document).ready(function () {
//     const vehicles = {
//         "2024": {
//             "Acura": ["ILX", "MDX", "NSX Type S", "RDX", "TLX"],
//             "Audi": ["A3", "A4", "A6", "A8", "Q3", "Q5", "Q7", "Q8", "e-tron GT", "RS3", "RS4 Avant", "RS5", "RS7", "R8"],
//             "BMW": ["2 Series", "3 Series", "4 Series", "5 Series", "7 Series", "8 Series", "X1", "X2", "X3", "X4", "X5", "X6", "X7", "Z4", "i4", "i7", "M2", "M3", "M4", "M5", "M8", "X5 M", "X6 M", "X7 M"],
//             "Buick": ["Enclave", "Encore", "Encore GX", "Envision", "Regal"],
//             "Cadillac": ["CT4", "CT5", "CT6", "Escalade", "XT4", "XT5", "XT6", "Lyriq"],
//             "Chevrolet": ["Bolt EV", "Bolt EUV", "Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["296 GTB", "812 Competizione", "SF90 Stradale", "Roma", "Purosangue"],
//             "Ford": ["Mustang Mach-E", "F-150", "Explorer", "Bronco", "Edge", "Escape"],
//             "Genesis": ["G70", "G80", "G90"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Kona", "Ioniq 5", "Ioniq 6", "Palisade", "Santa Cruz", "Santa Fe", "Tucson"],
//             "Infiniti": ["Q50", "Q60", "QX50", "QX55", "QX60", "QX80"],
//             "Jaguar": ["F-Type", "I-PACE", "XF", "XJ", "E-PACE"],
//             "Jeep": ["Cherokee", "Grand Cherokee", "Wrangler", "Gladiator", "Compass", "Renegade"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["CX-50", "CX-5", "CX-9", "Mazda3", "Mazda6", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class", "GLA", "GLB", "GLC", "GLE", "GLS", "EQB", "EQS"],
//             "MINI": ["Hardtop 2 Door", "Hardtop 4 Door", "Convertible", "Countryman", "Clubman", "Electric Cooper SE"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Nissan": ["Altima", "Ariya", "Frontier", "GT-R", "Kicks", "LEAF", "LEAF Plus", "Maxima", "Murano", "Pathfinder", "Rogue", "Sentra", "Titan", "Versa", "Z"],
//             "Polestar": ["Polestar 2", "Polestar 3", "Polestar 4"],
//             "Porsche": ["718 Cayman", "718 Boxster", "911 Carrera", "911 Turbo", "Panamera", "Taycan"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Rivian": ["R1T", "R1S", "R1T Adventure", "R1S Adventure"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "C-HR", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra", "Venza"],
//             "Volkswagen": ["Atlas", "Arteon", "Golf", "Jetta", "Passat", "Tiguan", "ID.4"],
//             "Volvo": ["XC40", "XC60", "XC90", "S60", "S90", "V60", "V90"]
//         },
//         "2023": {
//             "Acura": ["ILX", "MDX", "NSX Type S", "RDX", "TLX"],
//             "Audi": ["A3", "A4", "A6", "A8", "Q3", "Q5", "Q7", "Q8", "e-tron GT", "RS3", "RS4 Avant", "RS5", "RS7", "R8"],
//             "BMW": ["2 Series", "3 Series", "4 Series", "5 Series", "7 Series", "8 Series", "X1", "X2", "X3", "X4", "X5", "X6", "X7", "Z4", "i4", "i7", "M2", "M3", "M4", "M5", "M8", "X5 M", "X6 M", "X7 M"],
//             "Buick": ["Enclave", "Encore", "Encore GX", "Envision", "Regal"],
//             "Cadillac": ["CT4", "CT5", "CT6", "Escalade", "XT4", "XT5", "XT6", "Lyriq"],
//             "Chevrolet": ["Bolt EV", "Bolt EUV", "Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["296 GTB", "812 Competizione", "SF90 Stradale", "Roma", "Purosangue"],
//             "Ford": ["Mustang Mach-E", "F-150", "Explorer", "Bronco", "Edge", "Escape"],
//             "Genesis": ["G70", "G80", "G90"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Kona", "Ioniq 5", "Ioniq 6", "Palisade", "Santa Cruz", "Santa Fe", "Tucson"],
//             "Infiniti": ["Q50", "Q60", "QX50", "QX55", "QX60", "QX80"],
//             "Jaguar": ["F-Type", "I-PACE", "XF", "XJ", "E-PACE"],
//             "Jeep": ["Cherokee", "Grand Cherokee", "Wrangler", "Gladiator", "Compass", "Renegade"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["CX-50", "CX-5", "CX-9", "Mazda3", "Mazda6", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class", "GLA", "GLB", "GLC", "GLE", "GLS", "EQB", "EQS"],
//             "MINI": ["Hardtop 2 Door", "Hardtop 4 Door", "Convertible", "Countryman", "Clubman", "Electric Cooper SE"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Nissan": ["Altima", "Ariya", "Frontier", "GT-R", "Kicks", "LEAF", "LEAF Plus", "Maxima", "Murano", "Pathfinder", "Rogue", "Sentra", "Titan", "Versa", "Z"],
//             "Polestar": ["Polestar 2", "Polestar 3", "Polestar 4"],
//             "Porsche": ["718 Cayman", "718 Boxster", "911 Carrera", "911 Turbo", "Panamera", "Taycan"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Rivian": ["R1T", "R1S", "R1T Adventure", "R1S Adventure"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "C-HR", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra", "Venza"],
//             "Volkswagen": ["Atlas", "Arteon", "Golf", "Jetta", "Passat", "Tiguan", "ID.4"],
//             "Volvo": ["XC40", "XC60", "XC90", "S60", "S90", "V60", "V90"]
//         },
//         "2022": {
//             "Acura": ["ILX", "MDX", "NSX", "RDX", "TLX"],
//             "Audi": ["A3", "A4", "A6", "A8", "Q3", "Q5", "Q7", "Q8", "e-tron GT", "RS3", "RS4 Avant", "RS5", "RS7", "R8"],
//             "BMW": ["2 Series", "3 Series", "4 Series", "5 Series", "7 Series", "8 Series", "X1", "X2", "X3", "X4", "X5", "X6", "X7", "Z4", "i4", "i7", "M2", "M3", "M4", "M5", "M8", "X5 M", "X6 M", "X7 M"],
//             "Buick": ["Enclave", "Encore", "Encore GX", "Envision"],
//             "Cadillac": ["CT4", "CT5", "CT6", "Escalade", "XT4", "XT5", "XT6", "Lyriq"],
//             "Chevrolet": ["Bolt EV", "Bolt EUV", "Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["296 GTB", "812 Competizione", "SF90 Stradale", "Roma", "Portofino M"],
//             "Ford": ["Mustang Mach-E", "F-150", "Explorer", "Bronco", "Edge", "Escape"],
//             "Genesis": ["G70", "G80", "G90"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Kona", "Ioniq 5", "Ioniq 6", "Palisade", "Santa Cruz", "Santa Fe", "Tucson"],
//             "Infiniti": ["Q50", "Q60", "QX50", "QX55", "QX60", "QX80"],
//             "Jaguar": ["F-Type", "I-PACE", "XF", "XJ", "E-PACE"],
//             "Jeep": ["Cherokee", "Grand Cherokee", "Wrangler", "Gladiator", "Compass", "Renegade"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["CX-50", "CX-5", "CX-9", "Mazda3", "Mazda6", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class", "GLA", "GLB", "GLC", "GLE", "GLS", "EQB", "EQS"],
//             "MINI": ["Hardtop 2 Door", "Hardtop 4 Door", "Convertible", "Countryman", "Clubman", "Electric Cooper SE"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Nissan": ["Altima", "Ariya", "Frontier", "GT-R", "Kicks", "LEAF", "Maxima", "Murano", "Pathfinder", "Rogue", "Sentra", "Titan", "Versa", "Z"],
//             "Polestar": ["Polestar 2", "Polestar 1"],
//             "Porsche": ["718 Cayman", "718 Boxster", "911 Carrera", "911 Turbo", "Panamera", "Taycan"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Rivian": ["R1T", "R1S", "R1T Adventure", "R1S Adventure"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "C-HR", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra", "Venza"],
//             "Volkswagen": ["Atlas", "Arteon", "Golf", "Jetta", "Passat", "Tiguan", "ID.4"],
//             "Volvo": ["XC40", "XC60", "XC90", "S60", "S90", "V60", "V90"]
//         },
//         "2021": {
//             "Acura": ["ILX", "MDX", "RDX", "TLX"],
//             "Audi": ["A3", "A4", "A6", "A8", "Q3", "Q5", "Q7", "Q8", "e-tron", "RS3", "RS4 Avant", "RS5", "RS7", "R8"],
//             "BMW": ["2 Series", "3 Series", "4 Series", "5 Series", "7 Series", "8 Series", "X1", "X2", "X3", "X4", "X5", "X6", "X7", "Z4", "i4", "iX", "M2", "M3", "M4", "M5", "M8", "X5 M", "X6 M", "X7 M"],
//             "Buick": ["Enclave", "Encore", "Encore GX", "Envision"],
//             "Cadillac": ["CT4", "CT5", "CT6", "Escalade", "XT4", "XT5", "XT6", "Lyriq"],
//             "Chevrolet": ["Bolt EV", "Bolt EUV", "Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["296 GTB", "812 Competizione", "SF90 Stradale", "Roma", "Portofino M"],
//             "Ford": ["Mustang Mach-E", "F-150", "Explorer", "Bronco", "Edge", "Escape"],
//             "Genesis": ["G70", "G80", "G90"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Kona", "Ioniq 5", "Ioniq 6", "Palisade", "Santa Cruz", "Santa Fe", "Tucson"],
//             "Infiniti": ["Q50", "Q60", "QX50", "QX55", "QX60", "QX80"],
//             "Jaguar": ["F-Type", "I-PACE", "XF", "XJ", "E-PACE"],
//             "Jeep": ["Cherokee", "Grand Cherokee", "Wrangler", "Gladiator", "Compass", "Renegade"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["CX-5", "CX-9", "Mazda3", "Mazda6", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class", "GLA", "GLB", "GLC", "GLE", "GLS", "EQB", "EQS"],
//             "MINI": ["Hardtop 2 Door", "Hardtop 4 Door", "Convertible", "Countryman", "Clubman", "Electric Cooper SE"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Nissan": ["Altima", "Ariya", "Frontier", "GT-R", "Kicks", "LEAF", "Maxima", "Murano", "Pathfinder", "Rogue", "Sentra", "Titan", "Versa", "Z"],
//             "Polestar": ["Polestar 2", "Polestar 1"],
//             "Porsche": ["718 Cayman", "718 Boxster", "911 Carrera", "911 Turbo", "Panamera", "Taycan"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Rivian": ["R1T", "R1S"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "C-HR", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra", "Venza"],
//             "Volkswagen": ["Atlas", "Arteon", "Golf", "Jetta", "Passat", "Tiguan", "ID.4"],
//             "Volvo": ["XC40", "XC60", "XC90", "S60", "S90", "V60", "V90"]
//         },
//         "2020": {
//             "Acura": ["ILX", "MDX", "RDX", "TLX"],
//             "Audi": ["A3", "A4", "A6", "A8", "Q3", "Q5", "Q7", "Q8", "e-tron", "RS3", "RS4 Avant", "RS5", "RS7", "R8"],
//             "BMW": ["2 Series", "3 Series", "4 Series", "5 Series", "7 Series", "8 Series", "X1", "X2", "X3", "X4", "X5", "X6", "X7", "Z4", "i4", "iX", "M2", "M3", "M4", "M5", "M8", "X5 M", "X6 M", "X7 M"],
//             "Buick": ["Enclave", "Encore", "Encore GX", "Envision"],
//             "Cadillac": ["CT4", "CT5", "CT6", "Escalade", "XT4", "XT5", "XT6"],
//             "Chevrolet": ["Bolt EV", "Bolt EUV", "Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["F8 Tributo", "SF90 Stradale", "Roma", "Portofino M"],
//             "Ford": ["Mustang Mach-E", "F-150", "Explorer", "Bronco Sport", "Edge", "Escape"],
//             "Genesis": ["G70", "G80", "G90"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Kona", "Ioniq 5", "Palisade", "Santa Cruz", "Santa Fe", "Tucson"],
//             "Infiniti": ["Q50", "Q60", "QX50", "QX55", "QX60", "QX80"],
//             "Jaguar": ["F-Type", "I-PACE", "XF", "XJ", "E-PACE"],
//             "Jeep": ["Cherokee", "Grand Cherokee", "Wrangler", "Gladiator", "Compass", "Renegade"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["CX-5", "CX-9", "Mazda3", "Mazda6", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class", "GLA", "GLB", "GLC", "GLE", "GLS", "EQB", "EQS"],
//             "MINI": ["Hardtop 2 Door", "Hardtop 4 Door", "Convertible", "Countryman", "Clubman", "Electric Cooper SE"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Nissan": ["Altima", "Ariya", "Frontier", "GT-R", "Kicks", "LEAF", "Maxima", "Murano", "Pathfinder", "Rogue", "Sentra", "Titan", "Versa", "Z"],
//             "Polestar": ["Polestar 2", "Polestar 1"],
//             "Porsche": ["718 Cayman", "718 Boxster", "911 Carrera", "911 Turbo", "Panamera", "Taycan"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Rivian": ["R1T (Pre-production)", "R1S (Pre-production)"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "C-HR", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra", "Venza"],
//             "Volkswagen": ["Atlas", "Arteon", "Golf", "Jetta", "Passat", "Tiguan", "ID.4"],
//             "Volvo": ["XC40", "XC60", "XC90", "S60", "S90", "V60", "V90"]
//         },
//         "2019": {
//             "Acura": ["ILX", "MDX", "RDX", "TLX"],
//             "Audi": ["A3", "A4", "A6", "A8", "Q3", "Q5", "Q7", "Q8", "e-tron", "RS3", "RS4 Avant", "RS5", "RS7", "R8"],
//             "BMW": ["2 Series", "3 Series", "4 Series", "5 Series", "7 Series", "8 Series", "X1", "X2", "X3", "X4", "X5", "X6", "X7", "Z4", "i3", "i8", "M2", "M3", "M4", "M5", "M8", "X5 M", "X6 M", "X7 M"],
//             "Buick": ["Enclave", "Encore", "Encore GX", "Envision"],
//             "Cadillac": ["CT6", "Escalade", "XT4", "XT5", "XT6"],
//             "Chevrolet": ["Bolt EV", "Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["F8 Tributo", "812 Superfast", "Portofino", "488 Pista"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Bronco Sport", "Edge", "Escape"],
//             "Genesis": ["G70", "G80", "G90"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Kona", "Ioniq", "Palisade", "Santa Cruz", "Santa Fe", "Tucson"],
//             "Infiniti": ["Q50", "Q60", "QX50", "QX55", "QX60", "QX80"],
//             "Jaguar": ["F-Type", "I-PACE", "XF", "XJ", "E-PACE"],
//             "Jeep": ["Cherokee", "Grand Cherokee", "Wrangler", "Gladiator", "Compass", "Renegade"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["CX-5", "CX-9", "Mazda3", "Mazda6", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class", "GLA", "GLB", "GLC", "GLE", "GLS", "EQB", "EQS"],
//             "MINI": ["Hardtop 2 Door", "Hardtop 4 Door", "Convertible", "Countryman", "Clubman", "Electric Cooper SE"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Nissan": ["Altima", "GT-R", "Frontier", "Kicks", "LEAF", "Maxima", "Murano", "Pathfinder", "Rogue", "Sentra", "Titan", "Versa"],
//             "Polestar": ["Polestar 1"],
//             "Porsche": ["718 Cayman", "718 Boxster", "911 Carrera", "911 Turbo", "Panamera", "Taycan"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "C-HR", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra", "Venza"],
//             "Volkswagen": ["Atlas", "Arteon", "Golf", "Jetta", "Passat", "Tiguan"],
//             "Volvo": ["XC40", "XC60", "XC90", "S60", "S90", "V60", "V90"]
//         },
//         "2018": {
//             "Chrysler": ["Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["812 Superfast", "Portofino", "488 Pista"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "Genesis": ["G70", "G80", "G90"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Kona", "Ioniq", "Santa Fe", "Tucson"],
//             "Infiniti": ["Q50", "Q60", "QX50", "QX55", "QX60", "QX80"],
//             "Jaguar": ["F-Type", "I-PACE", "XF", "XJ", "E-PACE"],
//             "Jeep": ["Cherokee", "Grand Cherokee", "Wrangler", "Gladiator", "Compass", "Renegade"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["CX-5", "CX-9", "Mazda3", "Mazda6", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class", "GLA", "GLB", "GLC", "GLE", "GLS", "EQC"],
//             "MINI": ["Hardtop 2 Door", "Hardtop 4 Door", "Convertible", "Countryman", "Clubman", "Electric Cooper SE"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Nissan": ["Altima", "Frontier", "GT-R", "Kicks", "LEAF", "Maxima", "Murano", "Pathfinder", "Rogue", "Sentra", "Titan", "Versa"],
//             "Porsche": ["718 Cayman", "718 Boxster", "911 Carrera", "911 Turbo", "Panamera", "Taycan"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "C-HR", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra", "Venza"],
//             "Volkswagen": ["Atlas", "Golf", "Jetta", "Passat", "Tiguan"],
//             "Volvo": ["XC40", "XC60", "XC90", "S60", "S90", "V60", "V90"]
//         },
//         "2017": {
//             "Acura": ["ILX", "MDX", "RDX", "TLX"],
//             "Audi": ["A3", "A4", "A6", "A8", "Q3", "Q5", "Q7", "Q8", "RS3", "RS4 Avant", "RS5", "RS7", "R8"],
//             "BMW": ["2 Series", "3 Series", "4 Series", "5 Series", "7 Series", "8 Series", "X1", "X2", "X3", "X4", "X5", "X6", "X7", "Z4", "i3", "i8", "M2", "M3", "M4", "M5", "M6", "X5 M", "X6 M", "X7 M"],
//             "Buick": ["Enclave", "Encore", "Encore GX", "Envision"],
//             "Cadillac": ["CT6", "Escalade", "XT5"],
//             "Chevrolet": ["Bolt EV", "Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["812 Superfast", "GTC4Lusso", "488 GTB", "488 Spider"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "Genesis": ["G80", "G90"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Kona", "Ioniq", "Santa Fe", "Tucson"],
//             "Infiniti": ["Q50", "Q60", "QX50", "QX55", "QX60", "QX80"],
//             "Jaguar": ["F-Type", "I-PACE", "XF", "XJ", "E-PACE"],
//             "Jeep": ["Cherokee", "Grand Cherokee", "Wrangler", "Compass", "Renegade"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["CX-5", "CX-9", "Mazda3", "Mazda6", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class", "GLA", "GLB", "GLC", "GLE", "GLS", "EQC"],
//             "MINI": ["Hardtop 2 Door", "Hardtop 4 Door", "Convertible", "Countryman", "Clubman"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Nissan": ["Altima", "Armada", "Frontier", "GT-R", "Juke", "Kicks", "LEAF", "Maxima", "Murano", "Pathfinder", "Rogue", "Sentra", "Titan", "Versa"],
//             "Porsche": ["718 Cayman", "718 Boxster", "911 Carrera", "911 Turbo", "Panamera", "918 Spyder"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "C-HR", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra", "Venza"],
//             "Volkswagen": ["Golf", "Jetta", "Passat", "Tiguan"],
//             "Volvo": ["XC60", "XC90", "S60", "V60"]
//         },
//         "2016": {
//             "Acura": ["ILX", "MDX", "RDX", "TLX"],
//             "Audi": ["A3", "A4", "A6", "A8", "Q3", "Q5", "Q7", "RS3", "RS4 Avant", "RS5", "RS7", "R8"],
//             "BMW": ["2 Series", "3 Series", "4 Series", "5 Series", "7 Series", "8 Series", "X1", "X2", "X3", "X4", "X5", "X6", "X7", "Z4", "i3", "i8", "M2", "M3", "M4", "M5", "M6", "X5 M", "X6 M", "X7 M"],
//             "Buick": ["Enclave", "Encore", "Encore GX", "Envision"],
//             "Cadillac": ["CT6", "Escalade", "XT5"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["GTC4Lusso", "488 GTB", "488 Spider"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "Genesis": ["G80", "G90"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Kona", "Ioniq", "Santa Fe", "Tucson"],
//             "Infiniti": ["Q50", "Q60", "QX50", "QX60", "QX80"],
//             "Jaguar": ["F-Type", "I-PACE", "XF", "XJ", "E-PACE"],
//             "Jeep": ["Cherokee", "Grand Cherokee", "Wrangler", "Compass", "Renegade"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["CX-5", "CX-9", "Mazda3", "Mazda6", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class", "GLA", "GLB", "GLC", "GLE", "GLS", "EQC"],
//             "MINI": ["Hardtop 2 Door", "Hardtop 4 Door", "Convertible", "Countryman", "Clubman"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Nissan": ["Altima", "Armada", "Frontier", "GT-R", "Juke", "Kicks", "LEAF", "Maxima", "Murano", "Pathfinder", "Rogue", "Sentra", "Titan", "Versa"],
//             "Porsche": ["718 Cayman", "718 Boxster", "911 Carrera", "911 Turbo", "Panamera", "918 Spyder"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "C-HR", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra", "Venza"],
//             "Volkswagen": ["Golf", "Jetta", "Passat", "Tiguan"],
//             "Volvo": ["XC60", "XC90", "S60", "V60"]
//         },
//         "2015": {
//             "Acura": ["ILX", "MDX", "RDX", "TLX"],
//             "Audi": ["A3", "A4", "A6", "A8", "Q3", "Q5", "Q7", "RS3", "RS4 Avant", "RS5", "RS7", "R8"],
//             "BMW": ["2 Series", "3 Series", "4 Series", "5 Series", "7 Series", "X1", "X3", "X4", "X5", "X6", "Z4", "i3", "i8", "M3", "M4", "M5", "M6"],
//             "Buick": ["Enclave", "Encore", "Encore GX", "Envision"],
//             "Cadillac": ["CT6", "Escalade", "XT5"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["488 GTB", "488 Spider", "LaFerrari"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "Genesis": ["G80", "G90"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Kona", "Santa Fe", "Tucson"],
//             "Infiniti": ["Q50", "Q60", "QX50", "QX60", "QX80"],
//             "Jaguar": ["F-Type", "XE", "XF", "XJ", "F-PACE"],
//             "Jeep": ["Cherokee", "Grand Cherokee", "Wrangler", "Compass", "Renegade"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["CX-5", "CX-9", "Mazda3", "Mazda6", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class", "GLA", "GLB", "GLC", "GLE", "GLS", "EQC"],
//             "MINI": ["Hardtop 2 Door", "Hardtop 4 Door", "Convertible", "Countryman", "Clubman"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Nissan": ["Altima", "Armada", "Frontier", "GT-R", "Juke", "Kicks", "LEAF", "Maxima", "Murano", "Pathfinder", "Rogue", "Sentra", "Titan", "Versa"],
//             "Porsche": ["911 Carrera", "911 Turbo", "Panamera", "918 Spyder"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "C-HR", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra", "Venza"],
//             "Volkswagen": ["Golf", "Jetta", "Passat", "Tiguan"],
//             "Volvo": ["XC60", "XC90", "S60", "V60"]
//         },
//         "2014": {
//             "Acura": ["ILX", "MDX", "RDX", "TLX"],
//             "Audi": ["A3", "A4", "A6", "A8", "Q3", "Q5", "Q7", "RS5", "RS7", "R8"],
//             "BMW": ["2 Series", "3 Series", "4 Series", "5 Series", "7 Series", "X1", "X3", "X4", "X5", "X6", "Z4", "i3", "i8", "M3", "M5", "M6"],
//             "Buick": ["Enclave", "Encore", "Regal", "Verano"],
//             "Cadillac": ["CT6", "Escalade", "XT5"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["LaFerrari", "California T"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "Genesis": ["G80", "G90"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Kona", "Santa Fe", "Tucson"],
//             "Infiniti": ["Q50", "Q60", "QX50", "QX60", "QX80"],
//             "Jaguar": ["F-Type", "XE", "XF", "XJ", "F-PACE"],
//             "Jeep": ["Cherokee", "Grand Cherokee", "Wrangler", "Compass", "Renegade"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["CX-5", "CX-9", "Mazda3", "Mazda6", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class", "GLA", "GLB", "GLC", "GLE", "GLS"],
//             "MINI": ["Hardtop 2 Door", "Hardtop 4 Door", "Convertible", "Countryman", "Clubman"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Porsche": ["911 Carrera", "911 Turbo", "Panamera"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "C-HR", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra", "Venza"],
//             "Volkswagen": ["Golf", "Jetta", "Passat", "Tiguan"],
//             "Volvo": ["XC60", "XC90", "S60", "V60"]
//         },
//         "2013": {
//             "Acura": ["ILX", "MDX", "RDX"],
//             "Audi": ["A3", "A4", "A6", "A8", "Q3", "Q5", "Q7", "RS5", "RS7", "R8"],
//             "BMW": ["1 Series", "3 Series", "5 Series", "7 Series", "X1", "X3", "X5", "X6", "Z4", "i3", "i8", "M3", "M5", "M6"],
//             "Buick": ["Enclave", "Encore", "Regal", "Verano"],
//             "Cadillac": ["CT6", "Escalade", "XT5"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["LaFerrari", "F12berlinetta"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "Genesis": ["G80"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Kona", "Santa Fe", "Tucson"],
//             "Infiniti": ["G37", "Q50", "JX35 (later QX60)", "FX37 (later QX70)", "QX56 (later QX80)"],
//             "Jaguar": ["XF", "XJ"],
//             "Jeep": ["Grand Cherokee", "Wrangler", "Compass", "Patriot"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["CX-5", "CX-9", "Mazda3", "Mazda6", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class", "GLA", "GLB", "GLC", "GLE", "GLS"],
//             "MINI": ["Hardtop 2 Door", "Hardtop 4 Door", "Convertible", "Countryman", "Clubman"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Porsche": ["911 Carrera", "911 Turbo", "Panamera"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "C-HR", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra", "Venza"],
//             "Volkswagen": ["Golf", "Jetta", "Passat", "Tiguan"],
//             "Volvo": ["XC60", "XC90", "S60", "V60"]
//         },
//         "2012": {
//             "Acura": ["MDX", "RDX", "TLX (Concept)"],
//             "Audi": ["A3", "A4", "A6", "A8", "Q3", "Q5", "Q7", "RS5", "RS7", "R8"],
//             "BMW": ["1 Series", "3 Series", "5 Series", "7 Series", "X1", "X3", "X5", "X6", "Z4", "i3", "i8", "M3", "M5", "M6"],
//             "Buick": ["Enclave", "Encore", "Regal", "Verano"],
//             "Cadillac": ["CT6", "Escalade", "XT5"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["F12berlinetta", "California 30"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "Genesis": ["G80"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Kona", "Santa Fe", "Tucson"],
//             "Infiniti": ["G37", "JX35 (later QX60)", "FX35 (later QX70)", "QX56 (later QX80)"],
//             "Jaguar": ["XF", "XJ"],
//             "Jeep": ["Grand Cherokee", "Wrangler", "Compass", "Patriot"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["CX-5", "CX-9", "Mazda3", "Mazda6", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class", "GLA", "GLB", "GLC", "GLE"],
//             "MINI": ["Hardtop 2 Door", "Hardtop 4 Door", "Convertible", "Countryman", "Clubman"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Porsche": ["911 Carrera", "911 Turbo", "Panamera"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "C-HR", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra", "Venza"],
//             "Volkswagen": ["Golf", "Jetta", "Passat", "Tiguan"],
//             "Volvo": ["XC60", "XC90", "S60", "V60"]
//         },
//         "2011": {
//             "Acura": ["MDX", "RDX"],
//             "Audi": ["A3", "A4", "A6", "A8", "Q5", "Q7", "RS5", "RS6", "R8"],
//             "BMW": ["1 Series", "3 Series", "5 Series", "7 Series", "X1", "X3", "X5", "X6", "Z4", "i3", "i8", "M3", "M5", "M6"],
//             "Buick": ["Enclave", "Regal", "LaCrosse", "Verano"],
//             "Cadillac": ["CT6", "Escalade", "XT5"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["FF", "California 30"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "Genesis": ["G80"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Kona", "Santa Fe", "Tucson"],
//             "Infiniti": ["G25", "G37", "EX35", "FX35", "QX56"],
//             "Jaguar": ["XF", "XJ"],
//             "Jeep": ["Grand Cherokee", "Wrangler", "Compass", "Patriot"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["CX-7", "CX-9", "Mazda3", "Mazda6", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class", "GLA", "GLB", "GLC", "GLE"],
//             "MINI": ["Hardtop 2 Door", "Hardtop 4 Door", "Convertible", "Countryman", "Clubman"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Porsche": ["911 Carrera", "911 Turbo", "Panamera"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "C-HR", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra", "Venza"],
//             "Volkswagen": ["Golf", "Jetta", "Passat", "Tiguan"],
//             "Volvo": ["XC60", "XC90", "S60", "V60"]
//         },
//         "2010": {
//             "Acura": ["MDX", "RDX"],
//             "Audi": ["A3", "A4", "A6", "A8", "Q5", "Q7", "RS5", "R8"],
//             "BMW": ["1 Series", "3 Series", "5 Series", "7 Series", "X1", "X3", "X5", "X6", "Z4", "i3", "i8", "M3", "M5", "M6"],
//             "Buick": ["Enclave", "LaCrosse", "Regal", "Lucerne"],
//             "Cadillac": ["CT6", "Escalade", "XT5"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["California", "458 Italia"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "Genesis": ["G80"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Kona", "Santa Fe", "Tucson"],
//             "Infiniti": ["G37", "EX35", "FX35", "QX56"],
//             "Jaguar": ["XF", "XJ"],
//             "Jeep": ["Grand Cherokee", "Wrangler", "Compass", "Patriot"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["CX-7", "CX-9", "Mazda3", "Mazda6", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class", "GLA", "GLB", "GLC"],
//             "MINI": ["Hardtop 2 Door", "Hardtop 4 Door", "Convertible", "Countryman", "Clubman"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Porsche": ["911 Carrera", "911 Turbo", "Panamera"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "C-HR", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra", "Venza"],
//             "Volkswagen": ["Golf", "Jetta", "Passat", "Tiguan"],
//             "Volvo": ["XC60", "XC90", "S60", "V60"]
//         },
//         "2009": {
//             "Acura": ["MDX", "RDX"],
//             "Audi": ["A3", "A4", "A6", "A8", "Q5", "Q7", "RS6", "R8"],
//             "BMW": ["1 Series", "3 Series", "5 Series", "7 Series", "X1", "X3", "X5", "X6", "Z4", "i3", "i8", "M3", "M5", "M6"],
//             "Buick": ["Enclave", "LaCrosse", "Regal", "Lucerne"],
//             "Cadillac": ["CT6", "Escalade", "XT5"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["458 Italia", "California"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Kona", "Santa Fe", "Tucson"],
//             "Infiniti": ["G37", "EX35", "FX35", "QX56"],
//             "Jaguar": ["XF", "XJ"],
//             "Jeep": ["Grand Cherokee", "Wrangler", "Compass", "Patriot"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["CX-7", "CX-9", "Mazda3", "Mazda6", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class", "GLA", "GLB"],
//             "MINI": ["Hardtop 2 Door", "Hardtop 4 Door", "Convertible", "Countryman", "Clubman"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Porsche": ["911 Carrera", "911 Turbo", "Panamera"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "C-HR", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra", "Venza"],
//             "Volkswagen": ["Golf", "Jetta", "Passat", "Tiguan"],
//             "Volvo": ["XC90", "S60", "V70"]
//         },
//         "2008": {
//             "Acura": ["MDX", "RDX"],
//             "Audi": ["A3", "A4", "A6", "A8", "Q5", "Q7", "R8"],
//             "BMW": ["1 Series", "3 Series", "5 Series", "7 Series", "X1", "X3", "X5", "X6", "Z4", "i3", "i8", "M3", "M5", "M6"],
//             "Buick": ["Enclave", "LaCrosse", "Lucerne"],
//             "Cadillac": ["CT6", "Escalade", "STS", "CTS", "SRX"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["California", "430 Scuderia"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Kona", "Santa Fe", "Tucson"],
//             "Infiniti": ["G35", "EX35", "FX35", "QX56"],
//             "Jaguar": ["XF", "XJ"],
//             "Jeep": ["Grand Cherokee", "Wrangler", "Compass", "Patriot"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["CX-7", "CX-9", "Mazda3", "Mazda6", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class", "GLA", "GLB"],
//             "MINI": ["Hardtop 2 Door", "Hardtop 4 Door", "Convertible", "Countryman", "Clubman"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Porsche": ["911 Carrera", "911 Turbo", "Panamera"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "C-HR", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra", "Venza"],
//             "Volkswagen": ["Golf", "Jetta", "Passat", "Tiguan"],
//             "Volvo": ["XC90", "S60", "V70"]
//         },
//         "2007": {
//             "Acura": ["MDX", "RDX"],
//             "Audi": ["A3", "A4", "A6", "A8", "Q7", "R8"],
//             "BMW": ["1 Series", "3 Series", "5 Series", "7 Series", "X1", "X3", "X5", "X6", "Z4", "i3", "i8", "M3", "M5", "M6"],
//             "Buick": ["Enclave", "LaCrosse", "Lucerne"],
//             "Cadillac": ["CT6", "Escalade", "STS", "CTS", "SRX"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["430 Scuderia", "599 GTB Fiorano"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Kona", "Santa Fe", "Tucson"],
//             "Infiniti": ["G35", "EX35", "FX35", "QX56"],
//             "Jaguar": ["XK", "XJ"],
//             "Jeep": ["Grand Cherokee", "Wrangler", "Compass", "Patriot"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["CX-7", "CX-9", "Mazda3", "Mazda6", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class", "GLA"],
//             "MINI": ["Hardtop 2 Door", "Hardtop 4 Door", "Convertible", "Countryman", "Clubman"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Porsche": ["911 Carrera", "911 Turbo", "Panamera"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "C-HR", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra", "Venza"],
//             "Volkswagen": ["Golf", "Jetta", "Passat", "Tiguan"],
//             "Volvo": ["XC90", "S60", "V70"]
//         },
//         "2006": {
//             "Acura": ["MDX", "RDX"],
//             "Audi": ["A3", "A4", "A6", "A8", "Q7", "R8"],
//             "BMW": ["1 Series", "3 Series", "5 Series", "7 Series", "X3", "X5", "X6", "Z4"],
//             "Buick": ["LaCrosse", "Lucerne"],
//             "Cadillac": ["CT6", "Escalade", "STS", "CTS", "SRX"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["599 GTB Fiorano", "F430"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Kona", "Santa Fe", "Tucson"],
//             "Infiniti": ["G35", "EX35", "FX35", "QX56"],
//             "Jaguar": ["XK", "XJ"],
//             "Jeep": ["Grand Cherokee", "Wrangler", "Compass", "Liberty"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["MX-5 Miata", "Mazda3", "Mazda6", "CX-7"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class"],
//             "MINI": ["Hardtop 2 Door", "Hardtop 4 Door", "Convertible", "Countryman", "Clubman"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Porsche": ["911 Carrera", "911 Turbo", "Panamera"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "C-HR", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra", "Venza"],
//             "Volkswagen": ["Golf", "Jetta", "Passat", "Tiguan"],
//             "Volvo": ["XC90", "S60", "V70"]
//         },
//         "2005": {
//             "Acura": ["MDX", "RDX"],
//             "Audi": ["A3", "A4", "A6", "A8", "Q7", "R8"],
//             "BMW": ["1 Series", "3 Series", "5 Series", "7 Series", "X3", "X5", "Z4"],
//             "Buick": ["LaCrosse", "Rendezvous"],
//             "Cadillac": ["CT6", "Escalade", "STS", "CTS", "SRX"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["F430", "612 Scaglietti"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Kona", "Santa Fe", "Tucson"],
//             "Infiniti": ["G35", "FX35", "QX56"],
//             "Jaguar": ["XK", "XJ"],
//             "Jeep": ["Grand Cherokee", "Wrangler", "Liberty"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["Mazda3", "Mazda6", "MX-5 Miata", "CX-7"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class"],
//             "MINI": ["Hardtop 2 Door", "Hardtop 4 Door", "Convertible", "Countryman", "Clubman"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Porsche": ["911 Carrera", "911 Turbo", "Panamera"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "C-HR", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra", "Venza"],
//             "Volkswagen": ["Golf", "Jetta", "Passat", "Tiguan"],
//             "Volvo": ["XC90", "S60", "V70"]
//         },
//         "2004": {
//             "Acura": ["MDX", "TSX"],
//             "Audi": ["A3", "A4", "A6", "A8", "RS6", "Q7", "R8 (Concept)"],
//             "BMW": ["3 Series", "5 Series", "7 Series", "X3", "X5", "Z4"],
//             "Buick": ["Century", "LaCrosse", "Rendezvous"],
//             "Cadillac": ["CT6", "Escalade", "STS", "CTS", "SRX"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["612 Scaglietti", "360 Challenge Stradale"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Santa Fe", "Tucson"],
//             "Infiniti": ["G35", "FX35", "QX56"],
//             "Jaguar": ["XK", "XJ"],
//             "Jeep": ["Grand Cherokee", "Wrangler", "Liberty"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["Mazda3", "Mazda6", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class"],
//             "MINI": ["Hardtop 2 Door", "Hardtop 4 Door", "Convertible", "Countryman", "Clubman"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Porsche": ["911 Carrera", "911 Turbo", "Panamera"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "C-HR", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra", "Venza"],
//             "Volkswagen": ["Golf", "Jetta", "Passat", "Tiguan"],
//             "Volvo": ["XC90", "S60", "V70"]
//         },
//         "2003": {
//             "Acura": ["MDX", "TSX"],
//             "Audi": ["A3", "A4", "A6", "A8", "RS6", "Q7"],
//             "BMW": ["3 Series", "5 Series", "7 Series", "X5", "Z4"],
//             "Buick": ["Century", "Regal", "Rendezvous"],
//             "Cadillac": ["CT6", "Escalade", "STS", "CTS", "SRX"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["360 Challenge Stradale", "575M Maranello"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Santa Fe", "Tucson"],
//             "Infiniti": ["G35", "FX35", "QX4"],
//             "Jaguar": ["XK", "XJ"],
//             "Jeep": ["Grand Cherokee", "Wrangler", "Liberty"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["Mazda3", "Mazda6", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class"],
//             "MINI": ["Hardtop 2 Door", "Hardtop 4 Door", "Convertible", "Countryman", "Clubman"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Porsche": ["911 Carrera", "911 Turbo", "Panamera"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "C-HR", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra", "Venza"],
//             "Volkswagen": ["Golf", "Jetta", "Passat", "Tiguan"],
//             "Volvo": ["XC90", "S60", "V70"]
//         },
//         "2002": {
//             "Acura": ["MDX", "RSX"],
//             "Audi": ["A3", "A4", "A6", "A8", "RS6"],
//             "BMW": ["3 Series", "5 Series", "7 Series", "X5"],
//             "Buick": ["Century", "Regal", "Rendezvous"],
//             "Cadillac": ["CT6", "Escalade", "STS", "CTS", "SRX"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["575M Maranello", "360 Modena"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Santa Fe", "Tucson"],
//             "Infiniti": ["G20", "FX45", "QX4"],
//             "Jaguar": ["X-Type", "XK", "XJ"],
//             "Jeep": ["Grand Cherokee", "Wrangler", "Liberty"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["Mazda3", "Mazda6", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class"],
//             "MINI": ["Hardtop 2 Door", "Hardtop 4 Door", "Convertible", "Countryman", "Clubman"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Porsche": ["911 Carrera", "911 Turbo", "Panamera"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "C-HR", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra", "Venza"],
//             "Volkswagen": ["Golf", "Jetta", "Passat", "Tiguan"],
//             "Volvo": ["XC90", "S60", "V70"]
//         },
//         "2001": {
//             "Acura": ["MDX", "RSX"],
//             "Audi": ["A3", "A4", "A6", "A8", "RS4", "RS6"],
//             "BMW": ["3 Series", "5 Series", "7 Series"],
//             "Buick": ["Century", "Regal", "Rendezvous"],
//             "Cadillac": ["CT6", "Escalade", "STS", "CTS"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["360 Modena", "550 Maranello"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Santa Fe", "Tucson"],
//             "Infiniti": ["G20", "FX45", "QX4"],
//             "Jaguar": ["X-Type", "XK", "XJ"],
//             "Jeep": ["Grand Cherokee", "Wrangler", "Liberty"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["Mazda3", "Mazda6", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class"],
//             "MINI": ["Hardtop 2 Door", "Hardtop 4 Door", "Convertible", "Countryman", "Clubman"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Porsche": ["911 Carrera", "911 Turbo", "Panamera"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "C-HR", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra", "Venza"],
//             "Volkswagen": ["Golf", "Jetta", "Passat", "Tiguan"],
//             "Volvo": ["XC90", "S60", "V70"]
//         },
//         "2000": {
//             "Acura": ["MDX", "RSX"],
//             "Audi": ["A3", "A4", "A6", "A8", "RS4", "RS6"],
//             "BMW": ["3 Series", "5 Series"],
//             "Buick": ["Century", "Regal"],
//             "Cadillac": ["CT6", "Escalade", "STS", "CTS"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["550 Maranello", "360 Modena"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Santa Fe", "Tucson"],
//             "Infiniti": ["G20", "FX45", "QX4"],
//             "Jaguar": ["X-Type", "XK", "XJ"],
//             "Jeep": ["Grand Cherokee", "Wrangler", "Cherokee"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["Mazda6", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class"],
//             "MINI": ["Hardtop 2 Door", "Hardtop 4 Door", "Convertible", "Countryman", "Clubman"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Porsche": ["911 Carrera", "911 Turbo", "Panamera"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "C-HR", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra", "Venza"],
//             "Volkswagen": ["Golf", "Jetta", "Passat", "Tiguan"],
//             "Volvo": ["XC90", "S60", "V70"]
//         },
//         "1999": {
//             "Acura": ["MDX", "TL"],
//             "Audi": ["A3", "A4", "A6", "A8", "RS4"],
//             "BMW": ["3 Series"],
//             "Buick": ["Century", "Regal"],
//             "Cadillac": ["CT6", "Escalade", "STS", "CTS"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["360 Modena", "550 Maranello"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Santa Fe", "Tucson"],
//             "Infiniti": ["G20", "FX45", "QX4"],
//             "Jaguar": ["X-Type", "XK", "XJ"],
//             "Jeep": ["Grand Cherokee", "Wrangler", "Cherokee"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["626", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class"],
//             "MINI": ["Hardtop 2 Door", "Hardtop 4 Door", "Convertible", "Countryman", "Clubman"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Porsche": ["911 Carrera", "911 Turbo", "Panamera"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra"],
//             "Volkswagen": ["Beetle", "Golf", "Jetta", "Passat"],
//             "Volvo": ["S70", "V70", "XC70"]
//         },
//         "1998": {
//             "Acura": ["CL", "TL"],
//             "Audi": ["A3", "A4", "A6", "A8", "RS4"],
//             "BMW": ["3 Series"],
//             "Buick": ["Century", "Regal"],
//             "Cadillac": ["CT6", "Escalade", "STS", "CTS"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["550 Maranello", "F355"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Santa Fe", "Tucson"],
//             "Infiniti": ["G20", "FX45", "QX4"],
//             "Jaguar": ["XJ"],
//             "Jeep": ["Grand Cherokee", "Wrangler", "Cherokee"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["626", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Porsche": ["911 Carrera", "911 Turbo", "Panamera"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra"],
//             "Volkswagen": ["Beetle", "Golf", "Jetta", "Passat"],
//             "Volvo": ["S70", "V70", "XC70"]
//         },
//         "1997": {
//             "Acura": ["CL", "TL"],
//             "Audi": ["A3", "A4", "A6", "A8", "RS4"],
//             "BMW": ["3 Series"],
//             "Buick": ["Century", "Regal"],
//             "Cadillac": ["CT6", "Escalade", "STS", "CTS"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["F355", "456 M"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Santa Fe", "Tucson"],
//             "Infiniti": ["G20", "QX4"],
//             "Jaguar": ["XJ"],
//             "Jeep": ["Grand Cherokee", "Wrangler", "Cherokee"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["626", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Porsche": ["911 Carrera", "911 Turbo", "Panamera"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra"],
//             "Volkswagen": ["Beetle", "Golf", "Jetta", "Passat"],
//             "Volvo": ["S70", "V70", "XC70"]
//         },
//         "1996": {
//             "Acura": ["CL", "TL"],
//             "Audi": ["A3", "A4", "A6", "A8"],
//             "BMW": ["3 Series"],
//             "Buick": ["Century", "Regal"],
//             "Cadillac": ["CT6", "Escalade", "STS", "CTS"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["456 M", "F355"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Santa Fe", "Tucson"],
//             "Infiniti": ["G20", "QX4"],
//             "Jaguar": ["XJ"],
//             "Jeep": ["Grand Cherokee", "Wrangler", "Cherokee"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["626", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Porsche": ["911 Carrera", "911 Turbo", "Panamera"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra"],
//             "Volkswagen": ["Beetle", "Golf", "Jetta", "Passat"],
//             "Volvo": ["S70", "V70", "XC70"]
//         },
//         "1995": {
//             "Acura": ["CL", "TL"],
//             "Audi": ["A3", "A4", "A6", "A8"],
//             "Buick": ["Century", "Regal"],
//             "Cadillac": ["CT6", "Escalade", "STS", "CTS"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["F355", "456 GT"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Santa Fe", "Tucson"],
//             "Infiniti": ["G20", "QX4"],
//             "Jaguar": ["XJ"],
//             "Jeep": ["Grand Cherokee", "Wrangler", "Cherokee"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["626", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Porsche": ["911 Carrera", "911 Turbo", "Panamera"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra"],
//             "Volkswagen": ["Beetle", "Golf", "Jetta", "Passat"],
//             "Volvo": ["850", "850 Turbo", "850 R"]
//         },
//         "1994": {
//             "Acura": ["CL", "Legend"],
//             "Audi": ["A3", "A4", "A6", "A8"],
//             "Buick": ["Century", "Regal"],
//             "Cadillac": ["CT6", "Escalade", "STS", "CTS"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["456 GT", "F355"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Santa Fe", "Tucson"],
//             "Infiniti": ["G20", "QX4"],
//             "Jaguar": ["XJ"],
//             "Jeep": ["Grand Cherokee", "Wrangler", "Cherokee"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["626", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Porsche": ["911 Carrera", "911 Turbo", "Panamera"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra"],
//             "Volkswagen": ["Beetle", "Golf", "Jetta", "Passat"],
//             "Volvo": ["850", "850 Turbo", "850 R"]
//         },
//         "1993": {
//             "Acura": ["CL", "Legend"],
//             "Audi": ["A3", "A4", "A6", "A8"],
//             "Buick": ["Century", "Regal"],
//             "Cadillac": ["CT6", "Escalade", "STS", "CTS"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["F355", "348"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Santa Fe", "Tucson"],
//             "Infiniti": ["G20", "QX4"],
//             "Jaguar": ["XJ"],
//             "Jeep": ["Grand Cherokee", "Wrangler", "Cherokee"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["626", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Porsche": ["911 Carrera", "911 Turbo", "Panamera"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra"],
//             "Volkswagen": ["Beetle", "Golf", "Jetta", "Passat"],
//             "Volvo": ["850", "850 Turbo", "850 R"]
//         },
//         "1992": {
//             "Acura": ["Legend"],
//             "Audi": ["A3", "A4", "A6", "A8"],
//             "Buick": ["Century", "Regal"],
//             "Cadillac": ["CT6", "Escalade", "STS", "CTS"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["348", "F512 M"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Santa Fe", "Tucson"],
//             "Infiniti": ["G20", "QX4"],
//             "Jaguar": ["XJ"],
//             "Jeep": ["Grand Cherokee", "Wrangler", "Cherokee"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["626", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Porsche": ["911 Carrera", "911 Turbo", "Panamera"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra"],
//             "Volkswagen": ["Beetle", "Golf", "Jetta", "Passat"],
//             "Volvo": ["850", "850 Turbo", "850 R"]
//         },
//         "1991": {
//             "Acura": ["Legend"],
//             "Audi": ["A3", "A4", "A6", "A8"],
//             "Buick": ["Century", "Regal"],
//             "Cadillac": ["CT6", "Escalade", "STS", "CTS"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["F512 M", "348 TS"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Santa Fe", "Tucson"],
//             "Infiniti": ["G20", "QX4"],
//             "Jaguar": ["XJ"],
//             "Jeep": ["Grand Cherokee", "Wrangler", "Cherokee"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["626", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Porsche": ["911 Carrera", "911 Turbo", "Panamera"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra"],
//             "Volkswagen": ["Beetle", "Golf", "Jetta", "Passat"],
//             "Volvo": ["940", "740"]
//         },
//         "1990": {
//             "Acura": ["Legend"],
//             "Audi": ["80", "90", "V8", "100"],
//             "Buick": ["Century", "Regal"],
//             "Cadillac": ["CT6", "Escalade", "STS", "CTS"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["348 TS", "348 GTB"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Santa Fe", "Tucson"],
//             "Infiniti": ["Q45", "G20"],
//             "Jaguar": ["XJ"],
//             "Jeep": ["Grand Cherokee", "Wrangler", "Cherokee"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["626", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Porsche": ["911 Carrera", "911 Turbo", "Panamera"],
//             "RAM": ["1500", "2500", "3500", "ProMaster City", "ProMaster"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Avalon", "Camry", "Corolla", "Highlander", "Land Cruiser", "Prius", "RAV4", "Sequoia", "Sienna", "Tacoma", "Tundra"],
//             "Volkswagen": ["Beetle", "Golf", "Jetta", "Passat"],
//             "Volvo": ["740", "760", "940"]
//         },
//         "1989": {
//             "Acura": ["Legend"],
//             "Audi": ["80", "90", "V8"],
//             "Buick": ["Century", "Regal"],
//             "Cadillac": ["CT6", "Escalade", "STS", "CTS"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["348 GTB", "F40"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Santa Fe", "Tucson"],
//             "Jaguar": ["XJ6"],
//             "Jeep": ["Wagoneer"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["626", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Camry", "Corolla", "Land Cruiser", "Pickup"],
//             "Volkswagen": ["Golf", "Jetta", "Passat"],
//             "Volvo": ["240", "760"]
//         },
//         "1988": {
//             "Acura": ["Legend"],
//             "Audi": ["80", "90", "V8"],
//             "Buick": ["Century", "Regal"],
//             "Cadillac": ["CT6", "Escalade", "STS", "CTS"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["F40", "Testarossa"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Santa Fe", "Tucson"],
//             "Jaguar": ["XJ6"],
//             "Jeep": ["Wagoneer"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["626", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Camry", "Corolla", "Land Cruiser", "Pickup"],
//             "Volkswagen": ["Golf", "Jetta", "Passat"],
//             "Volvo": ["240", "760"]
//         },
//         "1987": {
//             "Acura": ["Legend"],
//             "Audi": ["80", "90", "V8"],
//             "Buick": ["Century", "Regal"],
//             "Cadillac": ["CT6", "Escalade", "STS", "CTS"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["Testarossa", "328 GTS"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Santa Fe", "Tucson"],
//             "Jaguar": ["XJ6"],
//             "Jeep": ["Wagoneer"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["626", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Camry", "Corolla", "Land Cruiser", "Pickup"],
//             "Volkswagen": ["Golf", "Jetta", "Passat"],
//             "Volvo": ["240", "760"]
//         },
//         "1988": {
//             "Acura": ["Legend"],
//             "Audi": ["80", "90", "V8"],
//             "Buick": ["Century", "Regal"],
//             "Cadillac": ["CT6", "Escalade", "STS", "CTS"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["F40", "Testarossa"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Santa Fe", "Tucson"],
//             "Jaguar": ["XJ6"],
//             "Jeep": ["Wagoneer"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["626", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Camry", "Corolla", "Land Cruiser", "Pickup"],
//             "Volkswagen": ["Golf", "Jetta", "Passat"],
//             "Volvo": ["240", "760"]
//         },
//         "1987": {
//             "Acura": ["Legend"],
//             "Audi": ["80", "90", "V8"],
//             "Buick": ["Century", "Regal"],
//             "Cadillac": ["CT6", "Escalade", "STS", "CTS"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["Testarossa", "328 GTS"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Santa Fe", "Tucson"],
//             "Jaguar": ["XJ6"],
//             "Jeep": ["Wagoneer"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["626", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Camry", "Corolla", "Land Cruiser", "Pickup"],
//             "Volkswagen": ["Golf", "Jetta", "Passat"],
//             "Volvo": ["240", "760"]
//         },
//         "1986": {
//             "Acura": ["Legend"],
//             "Audi": ["80", "90", "V8"],
//             "Buick": ["Century", "Regal"],
//             "Cadillac": ["CT6", "Escalade", "STS", "CTS"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["328 GTS", "328 GTB"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Santa Fe", "Tucson"],
//             "Jaguar": ["XJ6"],
//             "Jeep": ["Wagoneer"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["626", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Camry", "Corolla", "Land Cruiser", "Pickup"],
//             "Volkswagen": ["Golf", "Jetta", "Passat"],
//             "Volvo": ["240", "760"]
//         },
//         "1985": {
//             "Acura": ["Legend"],
//             "Audi": ["80", "90", "V8"],
//             "Buick": ["Century", "Regal"],
//             "Cadillac": ["CT6", "Escalade", "STS", "CTS"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["328 GTB", "288 GTO"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Santa Fe", "Tucson"],
//             "Jaguar": ["XJ6"],
//             "Jeep": ["Wagoneer"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["626", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Camry", "Corolla", "Land Cruiser", "Pickup"],
//             "Volkswagen": ["Golf", "Jetta", "Passat"],
//             "Volvo": ["240", "760"]
//         },
//         "1984": {
//             "Acura": ["Legend"],
//             "Audi": ["80", "90", "V8"],
//             "Buick": ["Century", "Regal"],
//             "Cadillac": ["CT6", "Escalade", "STS", "CTS"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["288 GTO", "Mondial QV"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Santa Fe", "Tucson"],
//             "Jaguar": ["XJ6"],
//             "Jeep": ["Wagoneer"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["626", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Camry", "Corolla", "Land Cruiser", "Pickup"],
//             "Volkswagen": ["Golf", "Jetta", "Passat"],
//             "Volvo": ["240", "760"]
//         },
//         "1983": {
//             "Acura": ["Legend"],
//             "Audi": ["80", "90", "V8"],
//             "Buick": ["Century", "Regal"],
//             "Cadillac": ["CT6", "Escalade", "STS", "CTS"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["Mondial QV"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Santa Fe", "Tucson"],
//             "Jaguar": ["XJ6"],
//             "Jeep": ["Wagoneer"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["626", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Camry", "Corolla", "Land Cruiser", "Pickup"],
//             "Volkswagen": ["Golf", "Jetta", "Passat"],
//             "Volvo": ["240", "760"]
//         },
//         "1982": {
//             "Acura": ["Legend"],
//             "Audi": ["80", "90", "V8"],
//             "Buick": ["Century", "Regal"],
//             "Cadillac": ["CT6", "Escalade", "STS", "CTS"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["Mondial 8"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Santa Fe", "Tucson"],
//             "Jaguar": ["XJ6"],
//             "Jeep": ["Wagoneer"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["626", "MX-5 Miata"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Camry", "Corolla", "Land Cruiser", "Pickup"],
//             "Volkswagen": ["Golf", "Jetta", "Passat"],
//             "Volvo": ["240", "760"]
//         },
//         "1981": {
//             "Acura": ["Legend"],
//             "Audi": ["80", "90"],
//             "Buick": ["Century", "Regal"],
//             "Cadillac": ["CT6", "Escalade", "STS", "CTS"],
//             "Chevrolet": ["Camaro", "Colorado", "Corvette", "Equinox", "Malibu", "Silverado", "Tahoe", "Traverse"],
//             "Chrysler": ["300", "Pacifica", "Voyager"],
//             "Dodge": ["Challenger", "Charger", "Durango", "Journey", "RAM 1500", "RAM 2500", "RAM 3500"],
//             "Ferrari": ["Mondial 8"],
//             "Ford": ["Mustang", "F-150", "Explorer", "Edge", "Escape"],
//             "GMC": ["Sierra 1500", "Canyon", "Terrain", "Acadia", "Yukon"],
//             "Honda": ["Accord", "Civic", "CR-V", "HR-V", "Pilot", "Passport", "Ridgeline"],
//             "Hyundai": ["Accent", "Elantra", "Santa Fe", "Tucson"],
//             "Jaguar": ["XJ6"],
//             "Jeep": ["Wagoneer"],
//             "Kia": ["Soul", "Seltos", "Sportage", "Sorento", "Telluride", "Niro"],
//             "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport"],
//             "Lexus": ["ES", "GS", "GX", "LC", "LS", "NX", "RX", "UX"],
//             "Lincoln": ["Aviator", "Corsair", "Navigator", "Nautilus", "Continental"],
//             "Maserati": ["Alfieri", "Ghibli", "Levante", "Quattroporte"],
//             "Mazda": ["626", "RX-7"],
//             "Mercedes-Benz": ["A-Class", "C-Class", "E-Class", "S-Class"],
//             "Mitsubishi": ["Outlander", "Eclipse Cross", "RVR", "Mirage"],
//             "Subaru": ["Ascent", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "WRX"],
//             "Toyota": ["4Runner", "Camry", "Corolla", "Land Cruiser", "Pickup"],
//             "Volkswagen": ["Golf", "Jetta", "Passat"],
//             "Volvo": ["240", "760"]
//         }
//     };

//     // Populate year dropdowns for both first and second vehicle
//     const years = Object.keys(vehicles).sort((a, b) => b - a);
//     years.forEach(year => {
//         $('#vehical_year, #vehicle-year-2').append(`<option value="${year}">${year}</option>`);
//     });

//     // Function to populate brand dropdown based on year
//     function populateBrandDropdown(year, brandSelect, modelSelect) {
//         brandSelect.empty().append('<option value="" selected disabled>Select Your Car Brand</option>');
//         modelSelect.empty().append('<option value="" selected disabled>Select Your Car Model</option>');

//         if (year && vehicles[year]) {
//             const brands = Object.keys(vehicles[year]);
//             brands.forEach(brand => {
//                 brandSelect.append(`<option value="${brand}">${brand}</option>`);
//             });
//         }
//     }

//     // Function to populate model dropdown based on brand
//     function populateModelDropdown(year, brand, modelSelect) {
//         modelSelect.empty().append('<option value="" selected disabled>Select Your Car Model</option>');

//         if (year && brand && vehicles[year][brand]) {
//             const models = vehicles[year][brand];
//             models.forEach(model => {
//                 modelSelect.append(`<option value="${model}">${model}</option>`);
//             });
//         }
//     }

//     // Event handlers for the first vehicle
//     $('#vehical_year').change(function () {
//         const year = $(this).val();
//         populateBrandDropdown(year, $('#vehicle_make'), $('#vehicle_make'));
//         $('#ownership-section').hide();
//     });

//     $('#vehicle_make').change(function () {
//         const year = $('#vehical_year').val();
//         const brand = $(this).val();
//         populateModelDropdown(year, brand, $('#vehicle_model'));
//         $('#ownership-section').hide();
//     });

//     $('#vehicle_model').change(function () {
//         const brand = $('#vehicle_make').val();
//         const model = $(this).val();
//         if (brand && model) {
//             $('#ownership-section').show();
//             $('#ownership-label').text(`Do you own or lease your ${brand} ${model}?`);
//         } else {
//             $('#ownership-section').hide();
//         }
//     });

//     // Event handlers for the second vehicle
//     // $('#vehicle-year-2').change(function () {
//     //     const year = $(this).val();
//     //     populateBrandDropdown(year, $('#vehicle-brand-2'), $('#vehicle-model-2'));
//     //     $('#ownership-section-2').hide();
//     // });

//     // $('#vehicle-brand-2').change(function () {
//     //     const year = $('#vehicle-year-2').val();
//     //     const brand = $(this).val();
//     //     populateModelDropdown(year, brand, $('#vehicle-model-2'));
//     //     $('#ownership-section-2').hide();
//     // });

//     // $('#vehicle-model-2').change(function () {
//     //     const brand = $('#vehicle-brand-2').val();
//     //     const model = $(this).val();
//     //     if (brand && model) {
//     //         $('#ownership-section-2').show();
//     //     } else {
//     //         $('#ownership-section-2').hide();
//     //     }
//     // });
// });

// $(document).ready(function() {
//     // Function to toggle the visibility of the incident details form
//     function toggleIncidentDetails() {
//         if ($('#incident-free-no').is(':checked')) {
//             $('#incident-details-form').show(); // Show the details form if "No" is selected
//             // Show the ticket count field if "No" is selected
//             $('.ticket-count-field').show();
//         } else {
//             $('#incident-details-form').hide(); // Hide the details form if "Yes" is selected
//         }
//     }

//     // Initial check to set the visibility based on the default checked radio button
//     toggleIncidentDetails();
    

//     // Attach event listener to the radio buttons
//     $('input[name="incident-free"]').change(toggleIncidentDetails);

//     // Function to increment the ticket count
//     $('#increment-btn').click(function() {
//         var ticketInput = $('#ticket-count');
//         var currentValue = parseInt(ticketInput.val());
//         ticketInput.val(currentValue + 1); // Increment the value by 1
//     });

//     // Function to decrement the ticket count
//     $('#decrement-btn').click(function() {
//         var ticketInput = $('#ticket-count');
//         var currentValue = parseInt(ticketInput.val());
//         if (currentValue > 0) {
//             ticketInput.val(currentValue - 1); // Decrement the value by 1 if greater than 0
//         }
//     });
// });

// $(document).ready(function() {
//     $('#prediction-form').on('submit', function(event) {
//         event.preventDefault();
//         const formData = $(this).serializeArray(); // Change to serializeArray for better control
//         const jsonData = {}; // Create an object to hold the form data

//         // Convert form data to JSON object
//         formData.forEach(function(item) {
//             jsonData[item.name] = item.value;
//         });

//         // Send the data as JSON
//         $.ajax({
//             url: '/predict',
//             type: 'POST',
//             contentType: 'application/json',
//             data: JSON.stringify(jsonData),
//             success: function(data) {
//                 $('#result').html('Predicted Premium: $' + data.predicted_premium);
//             },
//             error: function(xhr) {
//                 $('#result').html('Error: ' + xhr.responseJSON.error);
//             }
//         });
//     });
// });



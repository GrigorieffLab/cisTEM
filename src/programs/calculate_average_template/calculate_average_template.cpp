#include "../../core/core_headers.h"

class
        CalculateAverageTemplateApp : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(CalculateAverageTemplateApp)

// override the DoInteractiveUserInput

void CalculateAverageTemplateApp::DoInteractiveUserInput( ) {
    wxString input_reconstruction;
    wxString ouput_average_image;
    wxString output_std_image;
    bool     write_std_image = true;
    float    pixel_size      = 1;
    //	float		voltage_kV = 300.0;
    //	float		spherical_aberration_mm = 2.7;
    //	float		amplitude_contrast = 0.07;
    //	float		beam_tilt_x;
    //	float		beam_tilt_y;
    //	float		particle_shift_x;
    //	float		particle_shift_y;
    float    mask_radius = 100.0;
    float    padding     = 1.0;
    float    wanted_SNR  = 1.0;
    wxString my_symmetry = "C1";
    bool     apply_CTF;
    bool     apply_shifts;
    bool     apply_mask;
    bool     add_noise;
    float    angular_step;
    float    in_plane_angular_step;
    float    defocus_1, defocus_2;
    wxString input_search_images;

    int max_threads;

    UserInput* my_input = new UserInput("CalculateAverageTemplate", 1.0);

    input_reconstruction  = my_input->GetFilenameFromUser("Input reconstruction", "The 3D reconstruction from which projections are calculated", "my_reconstruction.mrc", true);
    input_search_images   = my_input->GetFilenameFromUser("Input images to be searched", "The input image stack, containing the images that should be searched", "image_stack.mrc", true);
    angular_step          = my_input->GetFloatFromUser("Out of plane angular step (0.0 = set automatically)", "Angular step size for global grid search", "0.0", 0.0);
    in_plane_angular_step = my_input->GetFloatFromUser("In plane angular step (0.0 = set automatically)", "Angular step size for in-plane rotations during the search", "0.0", 0.0);
    apply_CTF             = my_input->GetYesNoFromUser("Apply ctf?", "apply ctf to image?", "Yes");
    apply_shifts          = false;
    write_std_image       = my_input->GetYesNoFromUser("Write std image?", "Do you need to output std image of template?", "Yes");
    ouput_average_image   = my_input->GetFilenameFromUser("Output average image", "The output average image, containing the 2D projection", "my_projection_mean.mrc", false);
    if ( write_std_image == true )
        output_std_image = my_input->GetFilenameFromUser("Output std image", "The output std image, containing the 2D projection", "my_projection_std.mrc", false);
    pixel_size = my_input->GetFloatFromUser("Pixel size of reconstruction (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);

    //	voltage_kV = my_input->GetFloatFromUser("Beam energy (keV)", "The energy of the electron beam used to image the sample in kilo electron volts", "300.0", 0.0);
    //	spherical_aberration_mm = my_input->GetFloatFromUser("Spherical aberration (mm)", "Spherical aberration of the objective lens in millimeters", "2.7", 0.0);
    //	amplitude_contrast = my_input->GetFloatFromUser("Amplitude contrast", "Assumed amplitude contrast", "0.07", 0.0, 1.0);
    //	beam_tilt_x = my_input->GetFloatFromUser("Beam tilt along x [mrad]", "Beam tilt to be applied along the x axis in mrad", "0.0", -100.0, 100.0);
    //	beam_tilt_y = my_input->GetFloatFromUser("Beam tilt along y [mrad]", "Beam tilt to be applied along the y axis in mrad", "0.0", -100.0, 100.0);
    //	particle_shift_x = my_input->GetFloatFromUser("Particle shift along x (A)", "Average particle shift along the x axis as a result of beam tilt in A", "0.0", -1.0, 1.0);
    //	particle_shift_y = my_input->GetFloatFromUser("Particle shift along y (A)", "Average particle shift along the y axis as a result of beam tilt in A", "0.0", -1.0, 1.0);
    mask_radius = my_input->GetFloatFromUser("Mask radius (A)", "Radius of a circular mask to be applied to the final reconstruction in Angstroms", "100.0", 0.0);
    wanted_SNR  = my_input->GetFloatFromUser("Wanted SNR", "The ratio of signal to noise variance after adding Gaussian noise and before masking", "1.0", 0.0);
    padding     = my_input->GetFloatFromUser("Padding factor", "Factor determining how much the input volume is padded to improve projections", "1.0", 1.0);
    my_symmetry = my_input->GetSymmetryFromUser("Particle symmetry", "The assumed symmetry of the particle to be reconstructed", "C1");
    apply_mask  = my_input->GetYesNoFromUser("Apply mask", "Should the particles be masked with the circular mask?", "No");
    add_noise   = my_input->GetYesNoFromUser("Add noise", "Should the Gaussian noise be added?", "No");
    defocus_1   = my_input->GetFloatFromUser("Defocus 1", "defocus 1", "3000.0", 3000.0);
    defocus_2   = my_input->GetFloatFromUser("Defocus 2", "defocus 2", "3000.0", 3000.0);

#ifdef _OPENMP
    max_threads = my_input->GetIntFromUser("Max. threads to use for calculation", "When threading, what is the max threads to run", "1", 1);
#else
    max_threads = 1;
#endif

    delete my_input;

    //	my_current_job.Reset(14);
    my_current_job.ManualSetArguments("ttbtfffftbbbbffifft",
                                      input_reconstruction.ToUTF8( ).data( ),
                                      ouput_average_image.ToUTF8( ).data( ),
                                      write_std_image,
                                      output_std_image.ToUTF8( ).data( ),
                                      pixel_size,
                                      mask_radius,
                                      wanted_SNR,
                                      padding,
                                      my_symmetry.ToUTF8( ).data( ),
                                      apply_CTF,
                                      apply_shifts,
                                      apply_mask,
                                      add_noise,
                                      angular_step,
                                      in_plane_angular_step,
                                      max_threads,
                                      defocus_1,
                                      defocus_2,
                                      input_search_images);
}

// override the do calculation method which will be what is actually run..

bool CalculateAverageTemplateApp::DoCalculation( ) {
    wxString input_reconstruction         = my_current_job.arguments[0].ReturnStringArgument( );
    wxString ouput_average_image          = my_current_job.arguments[1].ReturnStringArgument( );
    bool     write_std_image              = my_current_job.arguments[2].ReturnBoolArgument( );
    wxString ouput_std_image              = my_current_job.arguments[3].ReturnStringArgument( );
    float    pixel_size                   = my_current_job.arguments[4].ReturnFloatArgument( );
    float    mask_radius                  = my_current_job.arguments[5].ReturnFloatArgument( );
    float    wanted_SNR                   = my_current_job.arguments[6].ReturnFloatArgument( );
    float    padding                      = my_current_job.arguments[7].ReturnFloatArgument( );
    wxString my_symmetry                  = my_current_job.arguments[8].ReturnStringArgument( );
    bool     apply_CTF                    = my_current_job.arguments[9].ReturnBoolArgument( );
    bool     apply_shifts                 = my_current_job.arguments[10].ReturnBoolArgument( );
    bool     apply_mask                   = my_current_job.arguments[11].ReturnBoolArgument( );
    bool     add_noise                    = my_current_job.arguments[12].ReturnBoolArgument( );
    float    angular_step                 = my_current_job.arguments[13].ReturnFloatArgument( );
    float    in_plane_angular_step        = my_current_job.arguments[14].ReturnFloatArgument( );
    int      max_threads                  = my_current_job.arguments[15].ReturnIntegerArgument( );
    float    defocus_1                    = my_current_job.arguments[16].ReturnFloatArgument( );
    float    defocus_2                    = my_current_job.arguments[17].ReturnFloatArgument( );
    wxString input_search_images_filename = my_current_job.arguments[18].ReturnStringArgument( );

    Image               projection_image;
    Image               final_image;
    Image               final_image_squared;
    Image               sum_image;
    Image               sum_of_square_image;
    ReconstructedVolume input_3d;
    Image               projection_3d;
    ImageFile           input_search_image_file;
    Image               input_image;
    Image               projection_filter;

    // input_search_image_file.OpenFile(input_search_images_filename.ToStdString( ), false);
    input_search_image_file.OpenFile("mature60S_DF5000A_z250nm_1.06Apix.mrc", false);
    input_image.ReadSlice(&input_search_image_file, 1);
    input_image.ReplaceOutliersWithMean(5.0f);
    input_image.ForwardFFT( );
    input_image.SwapRealSpaceQuadrants( );

    Curve whitening_filter;
    Curve number_of_terms;
    whitening_filter.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((input_image.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
    number_of_terms.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((input_image.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));

    input_image.ZeroCentralPixel( ); // equivalent to subtracting mean in real space
    input_image.Compute1DPowerSpectrumCurve(&whitening_filter, &number_of_terms);
    whitening_filter.SquareRoot( );
    whitening_filter.Reciprocal( );
    whitening_filter.MultiplyByConstant(1.0f / whitening_filter.ReturnMaximumValue( ));

    int image_counter                      = 0;
    int number_of_projections_to_calculate = 0;

    int   current_image;
    float average_score = 0.0;
    float average_sigma = 0.0;
    float variance;
    float mask_falloff = 10.0;

    float microscope_voltage_kv              = 300.0f;
    float microscope_spherical_aberration_mm = 2.7f;
    float amplitude_contrast                 = 0.07f;
    float defocus_angle                      = 0.0f;
    float phase_shift                        = 0.0f;
    float beam_tilt_x                        = 0.0f;
    float beam_tilt_y                        = 0.0f;
    float image_shift_x                      = 0.0f;
    float image_shift_y                      = 0.0f;

    float psi_step;

    wxArrayInt lines_to_process;

    cisTEMParameterLine input_parameters;

    cisTEMParameters input_star_file;

    EulerSearch  global_euler_search;
    ParameterMap parameter_map; // needed for euler search init
    parameter_map.SetAllTrue( );

    float psi_start = 0.0f;
    float psi_max   = 360.0f;
    // if ( in_plane_angular_step <= 0 ) {
    //     psi_step = rad_2_deg(pixel_size / mask_radius_search);
    //     psi_step = 360.0 / int(360.0 / psi_step + 0.5);
    // }
    // else {
    psi_step = in_plane_angular_step;
    // }
    int   best_parameters_to_keep      = 20;
    float high_resolution_limit_search = pixel_size * 2;
    global_euler_search.InitGrid(my_symmetry, angular_step, 0.0f, 0.0f, psi_max, psi_step, psi_start, pixel_size / high_resolution_limit_search, parameter_map, best_parameters_to_keep);

    if ( my_symmetry.StartsWith("C") ) // TODO 2x check me - w/o this O symm at least is broken
    {
        if ( global_euler_search.test_mirror == true ) // otherwise the theta max is set to 90.0 and test_mirror is set to true.  However, I don't want to have to test the mirrors.
        {
            global_euler_search.theta_max = 180.0f;
        }
    }

    global_euler_search.CalculateGridSearchPositions(false);

    int first_search_position = 0;
    int last_search_position  = global_euler_search.number_of_search_positions - 1;

    int   current_angular_position;
    float current_psi;
    // TODO unroll these loops and multiply the product.

    MRCFile input_file(input_reconstruction.ToStdString( ), false);
    MRCFile output_avg_file(ouput_average_image.ToStdString( ), true);

    AnglesAndShifts my_parameters;
    CTF             my_ctf;

    my_ctf.Init(microscope_voltage_kv, microscope_spherical_aberration_mm, amplitude_contrast, defocus_1, defocus_2, defocus_angle, 0.0, 0.0, 0.0, pixel_size, deg_2_rad(phase_shift));
    my_ctf.SetDefocus(defocus_1 / pixel_size, defocus_2 / pixel_size, deg_2_rad(defocus_angle));

    if ( (input_file.ReturnXSize( ) != input_file.ReturnYSize( )) || (input_file.ReturnXSize( ) != input_file.ReturnZSize( )) ) {
        MyPrintWithDetails("Error: Input reconstruction is not cubic\n");
        DEBUG_ABORT;
    }

    input_3d.InitWithDimensions(input_file.ReturnXSize( ), input_file.ReturnYSize( ), input_file.ReturnZSize( ), pixel_size, my_symmetry);
    input_3d.density_map->ReadSlices(&input_file, 1, input_3d.density_map->logical_z_dimension);
    //	input_3d.density_map->AddConstant(- input_3d.density_map->ReturnAverageOfRealValuesOnEdges());
    if ( padding != 1.0 ) {
        input_3d.density_map->Resize(input_3d.density_map->logical_x_dimension * padding, input_3d.density_map->logical_y_dimension * padding, input_3d.density_map->logical_z_dimension * padding, input_3d.density_map->ReturnAverageOfRealValuesOnEdges( ));
    }
    input_3d.mask_radius = mask_radius;
    input_3d.density_map->CorrectSinc(mask_radius / pixel_size);
    input_3d.PrepareForProjections(0.0, 2.0 * pixel_size);

    for ( current_angular_position = first_search_position; current_angular_position <= last_search_position; current_angular_position++ ) {
        //loop over each rotation angle
        for ( current_psi = psi_start; current_psi <= psi_max; current_psi += psi_step ) {
            number_of_projections_to_calculate++;
        }
    }

    int number_of_rotations = 0;
    for ( current_psi = psi_start; current_psi <= psi_max; current_psi += psi_step ) {
        number_of_rotations++;
    }
    ProgressBar* my_progress = new ProgressBar(number_of_projections_to_calculate);
    projection_3d.CopyFrom(input_3d.density_map);
    projection_image.Allocate(input_file.ReturnXSize( ), input_file.ReturnYSize( ), false);
    final_image.Allocate(input_file.ReturnXSize( ), input_file.ReturnYSize( ), true);
    final_image_squared.Allocate(input_file.ReturnXSize( ), input_file.ReturnYSize( ), true);
    sum_image.Allocate(input_file.ReturnXSize( ), input_file.ReturnYSize( ), true);
    sum_image.SetToConstant(0.0f);
    sum_of_square_image.Allocate(input_file.ReturnXSize( ), input_file.ReturnYSize( ), true);
    sum_of_square_image.SetToConstant(0.0f);
    projection_filter.Allocate(input_file.ReturnXSize( ), input_file.ReturnYSize( ), true);
    projection_filter.SetToConstant(0.0f);
    projection_filter.CalculateCTFImage(my_ctf);
    projection_filter.ApplyCurveFilter(&whitening_filter);

#pragma omp parallel for num_threads(max_threads) default(shared) private(current_angular_position, current_psi, input_parameters, my_parameters, variance) firstprivate(projection_image, final_image, final_image_squared)
    for ( current_angular_position = first_search_position; current_angular_position <= last_search_position; current_angular_position++ ) {
        RandomNumberGenerator local_random_generator(int(fabsf(global_random_number_generator.GetUniformRandom( ) * 50000)), true); // todo
        for ( current_psi = psi_start; current_psi <= psi_max; current_psi += psi_step ) {
            int view_counter = (current_angular_position - first_search_position) * int((psi_max - psi_start) / psi_step + 1) + int((current_psi - psi_start) / psi_step);

            wxPrintf("phi theta psi = %f %f %f\n", global_euler_search.list_of_search_parameters[current_angular_position][0], global_euler_search.list_of_search_parameters[current_angular_position][1], current_psi);
            my_parameters.Init(global_euler_search.list_of_search_parameters[current_angular_position][0], global_euler_search.list_of_search_parameters[current_angular_position][1], current_psi, 0.0, 0.0f);
            // my_ctf.Init(microscope_voltage_kv, microscope_spherical_aberration_mm, amplitude_contrast, defocus_1, defocus_2, defocus_angle, 0.0, 0.0, 0.0, pixel_size, phase_shift, -beam_tilt_x / 1000.0f, -beam_tilt_y / 1000.0f, -image_shift_x, -image_shift_y);
            projection_3d.ExtractSlice(projection_image, my_parameters);

            projection_image.SwapRealSpaceQuadrants( );
            if ( apply_CTF )
                projection_image.MultiplyPixelWise(projection_filter);
            projection_image.complex_values[0] = projection_3d.complex_values[0];
            projection_image.BackwardFFT( );

            final_image.CopyFrom(&projection_image);

            if ( add_noise && wanted_SNR != 0.0 ) {
                variance = final_image.ReturnVarianceOfRealValues( );
                final_image.AddGaussianNoise(sqrtf(variance / wanted_SNR), &local_random_generator);
            }

            if ( apply_mask )
                final_image.CosineMask(mask_radius / input_parameters.pixel_size, 6.0);
            final_image.QuickAndDirtyWriteSlice(wxString::Format("stack_%i.mrc", view_counter).ToStdString( ), 1);
            sum_image.AddImage(&final_image);
            if ( write_std_image ) {
                final_image.SquareRealValues( );
                sum_of_square_image.AddImage(&final_image);
            }
            image_counter++;
            my_progress->Update(image_counter);
        }
    }
    wxPrintf("number of projections %i\n", number_of_projections_to_calculate);
    wxPrintf("number of calculated images %i\n", image_counter);
    sum_image.DivideByConstant(image_counter);
    sum_image.WriteSlice(&output_avg_file, 1);
    if ( write_std_image ) {
        sum_of_square_image.DivideByConstant(image_counter);
        sum_of_square_image.SubtractSquaredImage(&sum_image);
        sum_of_square_image.SquareRootRealValues( );
        MRCFile output_std_file(ouput_std_image.ToStdString( ), true);
        sum_of_square_image.WriteSlice(&output_std_file, 1);
    }

    return true;
}

#include "../../core/core_headers.h"

class
        CalculateSNRRatioApp : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(CalculateSNRRatioApp)

// override the DoInteractiveUserInput

void CalculateSNRRatioApp::DoInteractiveUserInput( ) {
    wxString input_search_images;
    wxString input_reconstruction_1;
    wxString input_reconstruction_2;

    float pixel_size              = 1.0f;
    float voltage_kV              = 300.0f;
    float spherical_aberration_mm = 2.7f;
    float amplitude_contrast      = 0.07f;
    float defocus1                = 10000.0f;
    float defocus2                = 10000.0f;
    ;
    float    defocus_angle;
    float    phase_shift;
    float    high_resolution_limit   = 8.0;
    float    angular_step            = 5.0;
    float    angular_step_2          = 5.0;
    int      best_parameters_to_keep = 20;
    float    padding                 = 1.0;
    wxString my_symmetry             = "C1";
    float    in_plane_angular_step   = 0;
    float    in_plane_angular_step_2 = 0;

    UserInput* my_input = new UserInput("CalculateSNRRatio", 1.00);

    input_search_images     = my_input->GetFilenameFromUser("Input images to be searched", "The input image stack, containing the images that should be searched", "image_stack.mrc", true);
    input_reconstruction_1  = my_input->GetFilenameFromUser("First input template reconstruction", "The 3D reconstruction from which projections are calculated", "reconstruction.mrc", true);
    input_reconstruction_2  = my_input->GetFilenameFromUser("Second input template reconstruction", "The 3D reconstruction from which projections are calculated", "reconstruction.mrc", true);
    pixel_size              = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
    voltage_kV              = my_input->GetFloatFromUser("Beam energy (keV)", "The energy of the electron beam used to image the sample in kilo electron volts", "300.0", 0.0);
    spherical_aberration_mm = my_input->GetFloatFromUser("Spherical aberration (mm)", "Spherical aberration of the objective lens in millimeters", "2.7");
    amplitude_contrast      = my_input->GetFloatFromUser("Amplitude contrast", "Assumed amplitude contrast", "0.07", 0.0, 1.0);
    defocus1                = my_input->GetFloatFromUser("Defocus1 (angstroms)", "Defocus1 for the input image", "10000", 0.0);
    defocus2                = my_input->GetFloatFromUser("Defocus2 (angstroms)", "Defocus2 for the input image", "10000", 0.0);
    defocus_angle           = my_input->GetFloatFromUser("Defocus Angle (degrees)", "Defocus Angle for the input image", "0.0");
    phase_shift             = my_input->GetFloatFromUser("Phase Shift (degrees)", "Additional phase shift in degrees", "0.0");
    high_resolution_limit   = my_input->GetFloatFromUser("High resolution limit (A)", "High resolution limit of the data used for alignment in Angstroms", "8.0", 0.0);
    angular_step            = my_input->GetFloatFromUser("Out of plane angular step (0.0 = set automatically)", "Angular step size for global grid search", "0.0", 0.0);
    in_plane_angular_step   = my_input->GetFloatFromUser("In plane angular step (0.0 = set automatically)", "Angular step size for in-plane rotations during the search", "0.0", 0.0);
    padding                 = my_input->GetFloatFromUser("Padding factor", "Factor determining how much the input volume is padded to improve projections", "1.0", 1.0, 2.0);
    my_symmetry             = my_input->GetSymmetryFromUser("Template symmetry", "The symmetry of the template reconstruction", "C1");
    angular_step_2          = my_input->GetFloatFromUser("Inside loop out of plane angular step (0.0 = set automatically)", "Angular step size for global grid search", "0.0", 0.0);
    in_plane_angular_step_2 = my_input->GetFloatFromUser("In plane angular step (0.0 = set automatically)", "Angular step size for in-plane rotations during the search", "0.0", 0.0);

#ifdef ENABLEGPU
#endif

    int first_search_position  = -1;
    int last_search_position   = -1;
    int last_search_position_2 = -1;

    delete my_input;

    my_current_job.ManualSetArguments("tttfffffffffifftfiiffi",
                                      input_search_images.ToUTF8( ).data( ),
                                      input_reconstruction_1.ToUTF8( ).data( ),
                                      input_reconstruction_2.ToUTF8( ).data( ),
                                      pixel_size,
                                      voltage_kV,
                                      spherical_aberration_mm,
                                      amplitude_contrast,
                                      defocus1,
                                      defocus2,
                                      defocus_angle,
                                      high_resolution_limit,
                                      angular_step,
                                      best_parameters_to_keep,
                                      padding,
                                      phase_shift,
                                      my_symmetry.ToUTF8( ).data( ),
                                      in_plane_angular_step,
                                      first_search_position,
                                      last_search_position,
                                      angular_step_2,
                                      in_plane_angular_step_2,
                                      last_search_position_2);
}

// override the do calculation method which will be what is actually run..

bool CalculateSNRRatioApp::DoCalculation( ) {
    wxString input_search_images_filename    = my_current_job.arguments[0].ReturnStringArgument( );
    wxString input_reconstruction_1_filename = my_current_job.arguments[1].ReturnStringArgument( );
    wxString input_reconstruction_2_filename = my_current_job.arguments[2].ReturnStringArgument( );
    float    pixel_size                      = my_current_job.arguments[3].ReturnFloatArgument( );
    float    voltage_kV                      = my_current_job.arguments[4].ReturnFloatArgument( );
    float    spherical_aberration_mm         = my_current_job.arguments[5].ReturnFloatArgument( );
    float    amplitude_contrast              = my_current_job.arguments[6].ReturnFloatArgument( );
    float    defocus1                        = my_current_job.arguments[7].ReturnFloatArgument( );
    float    defocus2                        = my_current_job.arguments[8].ReturnFloatArgument( );
    float    defocus_angle                   = my_current_job.arguments[9].ReturnFloatArgument( );
    ;
    float    high_resolution_limit_search = my_current_job.arguments[10].ReturnFloatArgument( );
    float    angular_step                 = my_current_job.arguments[11].ReturnFloatArgument( );
    int      best_parameters_to_keep      = my_current_job.arguments[12].ReturnIntegerArgument( );
    float    padding                      = my_current_job.arguments[13].ReturnFloatArgument( );
    float    phase_shift                  = my_current_job.arguments[14].ReturnFloatArgument( );
    wxString my_symmetry                  = my_current_job.arguments[15].ReturnStringArgument( );
    float    in_plane_angular_step        = my_current_job.arguments[16].ReturnFloatArgument( );
    int      first_search_position        = my_current_job.arguments[17].ReturnIntegerArgument( );
    int      last_search_position         = my_current_job.arguments[18].ReturnIntegerArgument( );
    float    angular_step_2               = my_current_job.arguments[19].ReturnFloatArgument( );
    float    in_plane_angular_step_2      = my_current_job.arguments[20].ReturnFloatArgument( );
    int      last_search_position_2       = my_current_job.arguments[21].ReturnIntegerArgument( );

    int  padded_dimensions_x;
    int  padded_dimensions_y;
    int  pad_factor            = 6;
    int  number_of_rotations   = 0;
    int  number_of_rotations_2 = 0;
    long total_correlation_positions, total_correlation_positions_2;
    long current_correlation_position, current_correlation_position_2;
    long total_correlation_positions_per_thread;
    long pixel_counter;

    int          current_search_position, current_search_position_2;
    float        psi_step   = in_plane_angular_step;
    float        psi_step_2 = in_plane_angular_step_2;
    float        psi_max    = 360.0f;
    float        psi_start  = 0.0f;
    ParameterMap parameter_map; // needed for euler search init
    //for (int i = 0; i < 5; i++) {parameter_map[i] = true;}
    parameter_map.SetAllTrue( );
    float current_psi, current_psi_2;
    float variance;
    float ccs_one_view_sum            = 0.0;
    float ccs_one_view_sum_of_squares = 0.0;
    float ccs_one_view_var, ccs_one_view_max, ccs_one_view_max_scaled;
    float ccs_multi_views_sum                   = 0.0;
    float ccs_multi_views_sum_scaled            = 0.0;
    float ccs_multi_views_sum_of_squares        = 0.0;
    float ccs_multi_views_sum_of_squares_scaled = 0.0;

    Curve whitening_filter;
    Curve number_of_terms;

    Image           input_reconstruction_1, input_reconstruction_2, current_projection_1, current_projection_2;
    ImageFile       input_search_image_file;
    ImageFile       input_reconstruction_1_file, input_reconstruction_2_file;
    Image           input_image;
    EulerSearch     global_euler_search, global_euler_search_2;
    AnglesAndShifts angles;

    //wxString        output_histogram_file = "ccs.txt";
    //NumericTextFile histogram_file(output_histogram_file, OPEN_TO_WRITE, 1);

    input_search_image_file.OpenFile(input_search_images_filename.ToStdString( ), false);
    input_image.ReadSlice(&input_search_image_file, 1);

    input_reconstruction_1_file.OpenFile(input_reconstruction_1_filename.ToStdString( ), false);
    input_reconstruction_2_file.OpenFile(input_reconstruction_2_filename.ToStdString( ), false);
    input_reconstruction_1.ReadSlices(&input_reconstruction_1_file, 1, input_reconstruction_1_file.ReturnNumberOfSlices( ));
    input_reconstruction_2.ReadSlices(&input_reconstruction_2_file, 1, input_reconstruction_2_file.ReturnNumberOfSlices( ));

    // 1 is coarse search; 2 is finer TM search
    global_euler_search.InitGrid(my_symmetry, angular_step, 0.0f, 0.0f, psi_max, psi_step, psi_start, pixel_size / high_resolution_limit_search, parameter_map, best_parameters_to_keep);
    global_euler_search_2.InitGrid(my_symmetry, angular_step_2, 0.0f, 0.0f, psi_max, psi_step_2, psi_start, pixel_size / high_resolution_limit_search, parameter_map, best_parameters_to_keep);
    if ( my_symmetry.StartsWith("C") ) // TODO 2x check me - w/o this O symm at least is broken
    {
        if ( global_euler_search.test_mirror == true ) // otherwise the theta max is set to 90.0 and test_mirror is set to true.  However, I don't want to have to test the mirrors.
        {
            global_euler_search.theta_max   = 180.0f;
            global_euler_search_2.theta_max = 180.0f;
        }
    }
    global_euler_search.CalculateGridSearchPositions(false);
    global_euler_search_2.CalculateGridSearchPositions(false);

    whitening_filter.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((input_image.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
    number_of_terms.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((input_image.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));

    input_image.ReplaceOutliersWithMean(5.0f);
    input_image.ForwardFFT( );
    input_image.SwapRealSpaceQuadrants( );

    input_image.ZeroCentralPixel( );
    input_image.Compute1DPowerSpectrumCurve(&whitening_filter, &number_of_terms);
    whitening_filter.SquareRoot( );
    whitening_filter.Reciprocal( );
    whitening_filter.MultiplyByConstant(1.0f / whitening_filter.ReturnMaximumValue( ));

    total_correlation_positions    = 0;
    total_correlation_positions_2  = 0;
    current_correlation_position   = 0;
    current_correlation_position_2 = 0;

    // if running locally, search over all of them

    if ( is_running_locally == true ) {
        first_search_position  = 0;
        last_search_position   = global_euler_search.number_of_search_positions - 1;
        last_search_position_2 = global_euler_search_2.number_of_search_positions - 1;
    }

    // TODO unroll these loops and multiply the product.
    for ( current_search_position = first_search_position; current_search_position <= last_search_position; current_search_position++ ) {
        //loop over each rotation angle

        for ( current_psi = psi_start; current_psi <= psi_max; current_psi += psi_step ) {
            total_correlation_positions++;
        }
    }

    for ( current_psi = psi_start; current_psi <= psi_max; current_psi += psi_step ) {
        number_of_rotations++;
    }

    for ( current_search_position_2 = first_search_position; current_search_position_2 <= last_search_position_2; current_search_position_2++ ) {
        //loop over each rotation angle

        for ( current_psi_2 = psi_start; current_psi_2 <= psi_max; current_psi_2 += psi_step_2 ) {
            total_correlation_positions_2++;
        }
    }

    for ( current_psi_2 = psi_start; current_psi_2 <= psi_max; current_psi_2 += psi_step_2 ) {
        number_of_rotations_2++;
    }

    wxPrintf("Outside Loop:\n");
    wxPrintf("Searching %i positions on the Euler sphere (first-last: %i-%i)\n", last_search_position - first_search_position, first_search_position, last_search_position);
    wxPrintf("Searching %i rotations per position.\n", number_of_rotations);
    wxPrintf("There are %li correlation positions total.\n\n", total_correlation_positions);

    wxPrintf("Inside Loop:\n");
    wxPrintf("Searching %i positions on the Euler sphere (first-last: %i-%i)\n", last_search_position_2 - first_search_position, first_search_position, last_search_position_2);
    wxPrintf("Searching %i rotations per position.\n", number_of_rotations_2);
    wxPrintf("There are %li correlation positions total.\n\n", total_correlation_positions_2);

    CTF   input_ctf;
    Image projection_filter;
    projection_filter.Allocate(input_reconstruction_1_file.ReturnXSize( ), input_reconstruction_1_file.ReturnXSize( ), false);

    input_ctf.Init(voltage_kV, spherical_aberration_mm, amplitude_contrast, defocus1, defocus2, defocus_angle, 0.0, 0.0, 0.0, pixel_size, deg_2_rad(phase_shift));
    input_ctf.SetDefocus(defocus1 / pixel_size, defocus2 / pixel_size, deg_2_rad(defocus_angle));
    projection_filter.CalculateCTFImage(input_ctf);
    projection_filter.ApplyCurveFilter(&whitening_filter);

    //whitening_filter.WriteToFile("/tmp/filter.txt");
    //input_image.ApplyCurveFilter(&whitening_filter);
    //input_image.ZeroCentralPixel( );
    //input_image.DivideByConstant(sqrtf(input_image.ReturnSumOfSquares( )));

    current_projection_1.Allocate(input_reconstruction_1_file.ReturnXSize( ), input_reconstruction_1_file.ReturnXSize( ), false);
    current_projection_2.Allocate(input_reconstruction_2_file.ReturnXSize( ), input_reconstruction_2_file.ReturnXSize( ), false);

    input_reconstruction_1.ForwardFFT( );
    input_reconstruction_1.ZeroCentralPixel( );
    input_reconstruction_1.SwapRealSpaceQuadrants( );

    input_reconstruction_2.ForwardFFT( );
    input_reconstruction_2.ZeroCentralPixel( );
    input_reconstruction_2.SwapRealSpaceQuadrants( );
    for ( current_search_position = first_search_position; current_search_position <= last_search_position; current_search_position++ ) {
        for ( current_psi = psi_start; current_psi <= psi_max; current_psi += psi_step ) {
            angles.Init(global_euler_search.list_of_search_parameters[current_search_position][0], global_euler_search.list_of_search_parameters[current_search_position][1], current_psi, 0.0, 0.0);

            //TODO: padding; ratio inside of average; get different projection then tm search
            input_reconstruction_1.ExtractSlice(current_projection_1, angles, 1.0f, false);
            //current_projection_1.SwapRealSpaceQuadrants( );
            current_projection_1.MultiplyPixelWise(projection_filter);
            current_projection_1.BackwardFFT( );
            current_projection_1.AddConstant(-current_projection_1.ReturnAverageOfRealValuesOnEdges( ));
            current_projection_1.AddConstant(-current_projection_1.ReturnAverageOfRealValues( ));
            variance = current_projection_1.ReturnSumOfSquares( ) - powf(current_projection_1.ReturnAverageOfRealValues( ), 2);
            current_projection_1.DivideByConstant(sqrtf(variance));
            //current_projection_1.AddGaussianNoise(10.0f);
            current_projection_1.ForwardFFT( );
            // Zeroing the central pixel is probably not doing anything useful...
            current_projection_1.ZeroCentralPixel( );

            // extract projections from second 3d template
            input_reconstruction_2.ExtractSlice(current_projection_2, angles, 1.0f, false);
            //current_projection_2.SwapRealSpaceQuadrants( );
            current_projection_2.MultiplyPixelWise(projection_filter);
            current_projection_2.BackwardFFT( );
            current_projection_2.AddConstant(-current_projection_2.ReturnAverageOfRealValuesOnEdges( ));
            current_projection_2.AddConstant(-current_projection_2.ReturnAverageOfRealValues( ));
            variance = current_projection_2.ReturnSumOfSquares( ) - powf(current_projection_2.ReturnAverageOfRealValues( ), 2);
            current_projection_2.DivideByConstant(sqrtf(variance));
            //current_projection_2.AddGaussianNoise(10.0f);
            current_projection_2.ForwardFFT( );
            // Zeroing the central pixel is probably not doing anything useful...
            current_projection_2.ZeroCentralPixel( );

#ifdef MKL
            // Use the MKL
            vmcMulByConj(current_projection_2.real_memory_allocated / 2, reinterpret_cast<MKL_Complex8*>(current_projection_1.complex_values), reinterpret_cast<MKL_Complex8*>(current_projection_2.complex_values), reinterpret_cast<MKL_Complex8*>(current_projection_2.complex_values), VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
#else
            for ( pixel_counter = 0; pixel_counter < current_projection_2.real_memory_allocated / 2; pixel_counter++ ) {
                current_projection_2.complex_values[pixel_counter] = conj(current_projection_2.complex_values[pixel_counter]) * current_projection_1.complex_values[pixel_counter];
            }
#endif

            current_projection_2.SwapRealSpaceQuadrants( );
            current_projection_2.BackwardFFT( );
            ccs_one_view_max = current_projection_2.ReturnCentralPixelValue( );

            // then get avg. and std. from different orientations
            int i = 0;
            for ( current_search_position_2 = first_search_position; current_search_position_2 <= last_search_position_2; current_search_position_2++ ) {
                for ( current_psi_2 = psi_start; current_psi_2 <= psi_max; current_psi_2 += psi_step_2 ) {
                    if ( i != 0 ) {
                        angles.Init(global_euler_search_2.list_of_search_parameters[current_search_position_2][0], global_euler_search_2.list_of_search_parameters[current_search_position_2][1], current_psi_2, 0.0, 0.0);
                        // extract projections from second 3d template
                        input_reconstruction_2.ExtractSlice(current_projection_2, angles, 1.0f, false);
                        //current_projection_2.SwapRealSpaceQuadrants( );
                        current_projection_2.MultiplyPixelWise(projection_filter);
                        current_projection_2.BackwardFFT( );
                        current_projection_2.AddConstant(-current_projection_2.ReturnAverageOfRealValuesOnEdges( ));
                        current_projection_2.AddConstant(-current_projection_2.ReturnAverageOfRealValues( ));
                        variance = current_projection_2.ReturnSumOfSquares( ) - powf(current_projection_2.ReturnAverageOfRealValues( ), 2);
                        current_projection_2.DivideByConstant(sqrtf(variance));
                        //current_projection_2.AddGaussianNoise(10.0f);
                        current_projection_2.ForwardFFT( );
                        // Zeroing the central pixel is probably not doing anything useful...
                        current_projection_2.ZeroCentralPixel( );
#ifdef MKL
                        // Use the MKL
                        vmcMulByConj(current_projection_2.real_memory_allocated / 2, reinterpret_cast<MKL_Complex8*>(current_projection_1.complex_values), reinterpret_cast<MKL_Complex8*>(current_projection_2.complex_values), reinterpret_cast<MKL_Complex8*>(current_projection_2.complex_values), VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
#else
                        for ( pixel_counter = 0; pixel_counter < current_projection_2.real_memory_allocated / 2; pixel_counter++ ) {
                            current_projection_2.complex_values[pixel_counter] = conj(current_projection_2.complex_values[pixel_counter]) * current_projection_1.complex_values[pixel_counter];
                        }
#endif
                        current_projection_2.SwapRealSpaceQuadrants( );
                        current_projection_2.BackwardFFT( );
                        ccs_one_view_sum += current_projection_2.ReturnCentralPixelValue( );
                        ccs_one_view_sum_of_squares += powf(current_projection_2.ReturnCentralPixelValue( ), 2);
                    }
                    i++;
                }
            }
            ccs_one_view_sum /= total_correlation_positions_2 - 1; // avg of one view
            ccs_one_view_sum_of_squares /= total_correlation_positions_2 - 1; // sos of one view
            ccs_one_view_var        = ccs_one_view_sum_of_squares - powf(ccs_one_view_sum, 2);
            ccs_one_view_max_scaled = (ccs_one_view_max - ccs_one_view_sum) / sqrtf(ccs_one_view_var);

            ccs_multi_views_sum += ccs_one_view_max;
            ccs_multi_views_sum_of_squares += powf(ccs_one_view_max, 2);
            ccs_multi_views_sum_scaled += ccs_one_view_max_scaled;
            ccs_multi_views_sum_of_squares_scaled += powf(ccs_one_view_max_scaled, 2);
            //histogram_file.WriteCommentLine("%f", current_projection_1.ReturnCentralPixelValue( ));
        }
    }
    //histogram_file.Close( );
    ccs_multi_views_sum /= (float)total_correlation_positions;
    ccs_multi_views_sum_of_squares /= (float)total_correlation_positions;
    ccs_multi_views_sum_scaled /= (float)total_correlation_positions;
    ccs_multi_views_sum_of_squares_scaled /= (float)total_correlation_positions;
    float var_of_ccs        = ccs_multi_views_sum_of_squares - powf(ccs_multi_views_sum, 2);
    float var_of_ccs_scaled = ccs_multi_views_sum_of_squares_scaled - powf(ccs_multi_views_sum_scaled, 2);
    wxPrintf("Avg of ccs %f\n", ccs_multi_views_sum);
    wxPrintf("SD of ccs %f\n", sqrtf(var_of_ccs));
    wxPrintf("Avg of scaled ccs %f\n", ccs_multi_views_sum_scaled);
    wxPrintf("SD of scaled ccs %f\n", sqrtf(var_of_ccs_scaled));
    return true;
}

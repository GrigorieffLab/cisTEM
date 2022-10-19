#include "../../core/core_headers.h"

class
        CompareTemplateApp : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(CompareTemplateApp)

// override the DoInteractiveUserInput

void CompareTemplateApp::DoInteractiveUserInput( ) {
    wxString input_search_images;
    wxString input_reconstruction_particle_filename, input_reconstruction_correct_filename, input_reconstruction_wrong_filename;

    float pixel_size              = 1.0f;
    float voltage_kV              = 300.0f;
    float spherical_aberration_mm = 2.7f;
    float amplitude_contrast      = 0.07f;
    float defocus1                = 10000.0f;
    float defocus2                = 10000.0f;
    ;
    float    defocus_angle;
    float    phase_shift;
    float    high_resolution_limit          = 8.0;
    float    angular_step_sampling          = 5.0;
    float    angular_step_tm                = 5.0;
    int      best_parameters_to_keep        = 20;
    float    padding                        = 1.0;
    wxString my_symmetry                    = "C1";
    float    in_plane_angular_step_sampling = 0;
    float    in_plane_angular_step_tm       = 0;
    int      max_threads                    = 1;

    UserInput* my_input = new UserInput("CompareTemplate", 1.00);

    input_search_images                    = my_input->GetFilenameFromUser("Input images to be searched", "The input image stack, containing the images that should be searched", "image_stack.mrc", true);
    input_reconstruction_particle_filename = my_input->GetFilenameFromUser("Input template reconstruction for simulating particles", "The 3D reconstruction from which projections are calculated", "reconstruction.mrc", true);
    input_reconstruction_correct_filename  = my_input->GetFilenameFromUser("Correct input template reconstruction", "The 3D reconstruction from which projections are calculated", "reconstruction.mrc", true);
    input_reconstruction_wrong_filename    = my_input->GetFilenameFromUser("Wrong input template reconstruction", "The 3D reconstruction from which projections are calculated", "reconstruction.mrc", true);
    pixel_size                             = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
    voltage_kV                             = my_input->GetFloatFromUser("Beam energy (keV)", "The energy of the electron beam used to image the sample in kilo electron volts", "300.0", 0.0);
    spherical_aberration_mm                = my_input->GetFloatFromUser("Spherical aberration (mm)", "Spherical aberration of the objective lens in millimeters", "2.7");
    amplitude_contrast                     = my_input->GetFloatFromUser("Amplitude contrast", "Assumed amplitude contrast", "0.07", 0.0, 1.0);
    defocus1                               = my_input->GetFloatFromUser("Defocus1 (angstroms)", "Defocus1 for the input image", "10000", 0.0);
    defocus2                               = my_input->GetFloatFromUser("Defocus2 (angstroms)", "Defocus2 for the input image", "10000", 0.0);
    defocus_angle                          = my_input->GetFloatFromUser("Defocus Angle (degrees)", "Defocus Angle for the input image", "0.0");
    phase_shift                            = my_input->GetFloatFromUser("Phase Shift (degrees)", "Additional phase shift in degrees", "0.0");
    high_resolution_limit                  = my_input->GetFloatFromUser("High resolution limit (A)", "High resolution limit of the data used for alignment in Angstroms", "8.0", 0.0);
    angular_step_sampling                  = my_input->GetFloatFromUser("Sampling out of plane angular step (0.0 = set automatically)", "Angular step size for global grid search", "0.0", 0.0);
    in_plane_angular_step_sampling         = my_input->GetFloatFromUser("Sampling in plane angular step (0.0 = set automatically)", "Angular step size for in-plane rotations during the search", "0.0", 0.0);
    padding                                = my_input->GetFloatFromUser("Padding factor", "Factor determining how much the input volume is padded to improve projections", "1.0", 1.0, 2.0);
    my_symmetry                            = my_input->GetSymmetryFromUser("Template symmetry", "The symmetry of the template reconstruction", "C1");
    angular_step_tm                        = my_input->GetFloatFromUser("TM out of plane angular step (0.0 = set automatically)", "Angular step size for global grid search", "0.0", 0.0);
    in_plane_angular_step_tm               = my_input->GetFloatFromUser("TM in plane angular step (0.0 = set automatically)", "Angular step size for in-plane rotations during the search", "0.0", 0.0);
    max_threads                            = my_input->GetIntFromUser("Max threads", "threads used in openMP", "6", 1, 44);

#ifdef ENABLEGPU
#endif

    int first_search_position  = -1;
    int last_search_position   = -1;
    int last_search_position_2 = -1;

    delete my_input;

    my_current_job.ManualSetArguments("ttttfffffffffifftfiiffii",
                                      input_search_images.ToUTF8( ).data( ),
                                      input_reconstruction_particle_filename.ToUTF8( ).data( ),
                                      input_reconstruction_correct_filename.ToUTF8( ).data( ),
                                      input_reconstruction_wrong_filename.ToUTF8( ).data( ),
                                      pixel_size,
                                      voltage_kV,
                                      spherical_aberration_mm,
                                      amplitude_contrast,
                                      defocus1,
                                      defocus2,
                                      defocus_angle,
                                      high_resolution_limit,
                                      angular_step_sampling,
                                      best_parameters_to_keep,
                                      padding,
                                      phase_shift,
                                      my_symmetry.ToUTF8( ).data( ),
                                      in_plane_angular_step_sampling,
                                      first_search_position,
                                      last_search_position,
                                      angular_step_tm,
                                      in_plane_angular_step_tm,
                                      last_search_position_2,
                                      max_threads);
}

// override the do calculation method which will be what is actually run..

bool CompareTemplateApp::DoCalculation( ) {
    wxString input_search_images_filename           = my_current_job.arguments[0].ReturnStringArgument( );
    wxString input_reconstruction_particle_filename = my_current_job.arguments[1].ReturnStringArgument( );
    wxString input_reconstruction_correct_filename  = my_current_job.arguments[2].ReturnStringArgument( );
    wxString input_reconstruction_wrong_filename    = my_current_job.arguments[3].ReturnStringArgument( );
    float    pixel_size                             = my_current_job.arguments[4].ReturnFloatArgument( );
    float    voltage_kV                             = my_current_job.arguments[5].ReturnFloatArgument( );
    float    spherical_aberration_mm                = my_current_job.arguments[6].ReturnFloatArgument( );
    float    amplitude_contrast                     = my_current_job.arguments[7].ReturnFloatArgument( );
    float    defocus1                               = my_current_job.arguments[8].ReturnFloatArgument( );
    float    defocus2                               = my_current_job.arguments[9].ReturnFloatArgument( );
    float    defocus_angle                          = my_current_job.arguments[10].ReturnFloatArgument( );
    ;
    float    high_resolution_limit_search   = my_current_job.arguments[11].ReturnFloatArgument( );
    float    angular_step_sampling          = my_current_job.arguments[12].ReturnFloatArgument( );
    int      best_parameters_to_keep        = my_current_job.arguments[13].ReturnIntegerArgument( );
    float    padding                        = my_current_job.arguments[14].ReturnFloatArgument( );
    float    phase_shift                    = my_current_job.arguments[15].ReturnFloatArgument( );
    wxString my_symmetry                    = my_current_job.arguments[16].ReturnStringArgument( );
    float    in_plane_angular_step_sampling = my_current_job.arguments[17].ReturnFloatArgument( );
    int      first_search_position          = my_current_job.arguments[18].ReturnIntegerArgument( );
    int      last_search_position           = my_current_job.arguments[19].ReturnIntegerArgument( );
    float    angular_step_tm                = my_current_job.arguments[20].ReturnFloatArgument( );
    float    in_plane_angular_step_tm       = my_current_job.arguments[21].ReturnFloatArgument( );
    int      last_search_position_2         = my_current_job.arguments[22].ReturnIntegerArgument( );
    int      max_threads                    = my_current_job.arguments[23].ReturnIntegerArgument( );

    int  padded_dimensions_x;
    int  padded_dimensions_y;
    int  pad_factor                       = 6;
    int  number_of_rotations_sampled_view = 0;
    int  number_of_rotations_tm           = 0;
    long total_correlation_positions_sampled_view, total_correlation_positions_tm;
    long current_correlation_position_sampled_view, current_correlation_position_tm;
    long total_correlation_positions_per_thread;
    long pixel_counter;

    int          current_search_position_sampled_view, current_search_position_tm;
    float        psi_step_sampled_view = in_plane_angular_step_sampling;
    float        psi_step_tm           = in_plane_angular_step_tm;
    float        psi_max               = 360.0f;
    float        psi_start             = 0.0f;
    ParameterMap parameter_map; // needed for euler search init
    //for (int i = 0; i < 5; i++) {parameter_map[i] = true;}
    parameter_map.SetAllTrue( );
    float current_psi_sampled_view, current_psi_tm;
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

    Image           input_reconstruction_particle, input_reconstruction_correct, input_reconstruction_wrong, current_projection_image, current_projection_other, current_projection_correct_template;
    ImageFile       input_search_image_file;
    ImageFile       input_reconstruction_particle_file, input_reconstruction_correct_template_file, input_reconstruction_wrong_template_file;
    Image           input_image;
    EulerSearch     global_euler_search_sampled_view, global_euler_search_tm;
    AnglesAndShifts angles_sampled_view, angles_tm;

    //NumericTextFile output_file("clean_image_from_1_output.txt", OPEN_TO_WRITE, 3);

    //wxString        output_histogram_file = "ccs.txt";
    //NumericTextFile histogram_file(output_histogram_file, OPEN_TO_WRITE, 1);

    input_search_image_file.OpenFile(input_search_images_filename.ToStdString( ), false);
    input_image.ReadSlice(&input_search_image_file, 1);
    input_reconstruction_particle_file.OpenFile(input_reconstruction_particle_filename.ToStdString( ), false);
    input_reconstruction_correct_template_file.OpenFile(input_reconstruction_correct_filename.ToStdString( ), false);
    input_reconstruction_wrong_template_file.OpenFile(input_reconstruction_wrong_filename.ToStdString( ), false);
    input_reconstruction_particle.ReadSlices(&input_reconstruction_particle_file, 1, input_reconstruction_particle_file.ReturnNumberOfSlices( )); // particle in image
    input_reconstruction_correct.ReadSlices(&input_reconstruction_correct_template_file, 1, input_reconstruction_correct_template_file.ReturnNumberOfSlices( )); // correct template
    input_reconstruction_wrong.ReadSlices(&input_reconstruction_wrong_template_file, 1, input_reconstruction_wrong_template_file.ReturnNumberOfSlices( )); // wrong template

    // 1 is coarse search; 2 is finer TM search
    global_euler_search_sampled_view.InitGrid(my_symmetry, angular_step_sampling, 0.0f, 0.0f, psi_max, psi_step_sampled_view, psi_start, pixel_size / high_resolution_limit_search, parameter_map, best_parameters_to_keep);
    global_euler_search_tm.InitGrid(my_symmetry, angular_step_tm, 0.0f, 0.0f, psi_max, psi_step_tm, psi_start, pixel_size / high_resolution_limit_search, parameter_map, best_parameters_to_keep);
    if ( my_symmetry.StartsWith("C") ) // TODO 2x check me - w/o this O symm at least is broken
    {
        if ( global_euler_search_sampled_view.test_mirror == true ) // otherwise the theta max is set to 90.0 and test_mirror is set to true.  However, I don't want to have to test the mirrors.
        {
            global_euler_search_sampled_view.theta_max = 180.0f;
            global_euler_search_tm.theta_max           = 180.0f;
        }
    }
    global_euler_search_sampled_view.CalculateGridSearchPositions(false);
    global_euler_search_tm.CalculateGridSearchPositions(false);

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

    total_correlation_positions_sampled_view  = 0;
    total_correlation_positions_tm            = 0;
    current_correlation_position_sampled_view = 0;
    current_correlation_position_tm           = 0;

    // if running locally, search over all of them

    if ( is_running_locally == true ) {
        first_search_position  = 0;
        last_search_position   = global_euler_search_sampled_view.number_of_search_positions - 1;
        last_search_position_2 = global_euler_search_tm.number_of_search_positions - 1;
    }

    // TODO unroll these loops and multiply the product.
    for ( current_search_position_sampled_view = first_search_position; current_search_position_sampled_view <= last_search_position; current_search_position_sampled_view++ ) {
        //loop over each rotation angle

        for ( current_psi_sampled_view = psi_start; current_psi_sampled_view <= psi_max; current_psi_sampled_view += psi_step_sampled_view ) {
            total_correlation_positions_sampled_view++;
        }
    }

    for ( current_psi_sampled_view = psi_start; current_psi_sampled_view <= psi_max; current_psi_sampled_view += psi_step_sampled_view ) {
        number_of_rotations_sampled_view++;
    }

    for ( current_search_position_tm = first_search_position; current_search_position_tm <= last_search_position_2; current_search_position_tm++ ) {
        //loop over each rotation angle

        for ( current_psi_tm = psi_start; current_psi_tm <= psi_max; current_psi_tm += psi_step_tm ) {
            total_correlation_positions_tm++;
        }
    }

    for ( current_psi_tm = psi_start; current_psi_tm <= psi_max; current_psi_tm += psi_step_tm ) {
        number_of_rotations_tm++;
    }

    wxPrintf("Outside Loop - sampled views in an image:\n");
    wxPrintf("Searching %i positions on the Euler sphere (first-last: %i-%i)\n", last_search_position - first_search_position, first_search_position, last_search_position);
    wxPrintf("Searching %i rotations per position.\n", number_of_rotations_sampled_view);
    wxPrintf("There are %li correlation positions total.\n\n", total_correlation_positions_sampled_view);

    wxPrintf("Inside Loop - tm rotations:\n");
    wxPrintf("Searching %i positions on the Euler sphere (first-last: %i-%i)\n", last_search_position_2 - first_search_position, first_search_position, last_search_position_2);
    wxPrintf("Searching %i rotations per position.\n", number_of_rotations_tm);
    wxPrintf("There are %li correlation positions total.\n\n", total_correlation_positions_tm);

    // arrays for collecting data
    double* collected_aligned_ratio_data     = new double[total_correlation_positions_sampled_view];
    double* collected_ac_data                = new double[total_correlation_positions_sampled_view];
    double* collected_cc_data                = new double[total_correlation_positions_sampled_view];
    double* collected_ac_sum_data            = new double[total_correlation_positions_sampled_view];
    double* collected_cc_sum_data            = new double[total_correlation_positions_sampled_view];
    double* collected_ac_sum_of_squares_data = new double[total_correlation_positions_sampled_view];
    double* collected_cc_sum_of_squares_data = new double[total_correlation_positions_sampled_view];
    //double collected_max_data[total_correlation_positions];
    // double collected_avg_data[total_correlation_positions];
    //double collected_var_data[total_correlation_positions];

    float cc_val, ac_val, snr_ratio, pure_noise_cc_val;
    bool  is_maximum;

    for ( int line_counter = 0; line_counter < total_correlation_positions_sampled_view; line_counter++ ) {
        collected_ac_data[line_counter] = -10000.0;
        collected_cc_data[line_counter] = -10000.0;
    }
    ZeroDoubleArray(collected_ac_sum_data, total_correlation_positions_sampled_view);
    ZeroDoubleArray(collected_cc_sum_data, total_correlation_positions_sampled_view);
    ZeroDoubleArray(collected_ac_sum_of_squares_data, total_correlation_positions_sampled_view);
    ZeroDoubleArray(collected_cc_sum_of_squares_data, total_correlation_positions_sampled_view);
    ZeroDoubleArray(collected_aligned_ratio_data, total_correlation_positions_sampled_view);

    CTF   input_ctf;
    Image projection_filter;
    projection_filter.Allocate(input_reconstruction_particle_file.ReturnXSize( ), input_reconstruction_particle_file.ReturnXSize( ), false);

    input_ctf.Init(voltage_kV, spherical_aberration_mm, amplitude_contrast, defocus1, defocus2, defocus_angle, 0.0, 0.0, 0.0, pixel_size, deg_2_rad(phase_shift));
    input_ctf.SetDefocus(defocus1 / pixel_size, defocus2 / pixel_size, deg_2_rad(defocus_angle));
    projection_filter.CalculateCTFImage(input_ctf);
    projection_filter.ApplyCurveFilter(&whitening_filter);

    //whitening_filter.WriteToFile("/tmp/filter.txt");
    //input_image.ApplyCurveFilter(&whitening_filter);
    //input_image.ZeroCentralPixel( );
    //input_image.DivideByConstant(sqrtf(input_image.ReturnSumOfSquares( )));

    current_projection_image.Allocate(input_reconstruction_particle_file.ReturnXSize( ), input_reconstruction_particle_file.ReturnXSize( ), 1, false);
    current_projection_other.Allocate(input_reconstruction_wrong_template_file.ReturnXSize( ), input_reconstruction_wrong_template_file.ReturnXSize( ), 1, false);
    current_projection_correct_template.Allocate(input_reconstruction_correct_template_file.ReturnXSize( ), input_reconstruction_correct_template_file.ReturnXSize( ), 1, false);
    //pure_noise.Allocate(input_reconstruction_1_file.ReturnXSize( ), input_reconstruction_1_file.ReturnXSize( ), input_reconstruction_1_file.ReturnXSize( ), true);
    //pure_noise_2d.Allocate(input_reconstruction_1_file.ReturnXSize( ), input_reconstruction_1_file.ReturnXSize( ), false);
    //pure_noise.SetToConstant(0.0f);
    //pure_noise.AddGaussianNoise(5.0f);

    input_reconstruction_particle.ForwardFFT( );
    input_reconstruction_particle.ZeroCentralPixel( );
    input_reconstruction_particle.SwapRealSpaceQuadrants( );

    input_reconstruction_correct.ForwardFFT( );
    input_reconstruction_correct.ZeroCentralPixel( );
    input_reconstruction_correct.SwapRealSpaceQuadrants( );

    input_reconstruction_wrong.ForwardFFT( );
    input_reconstruction_wrong.ZeroCentralPixel( );
    input_reconstruction_wrong.SwapRealSpaceQuadrants( );

    //pure_noise.ForwardFFT( );
    //pure_noise.ZeroCentralPixel( );
    //pure_noise.SwapRealSpaceQuadrants( );

    // double temp_double_array[2];

    // step one: generate N projections from template 2 that is different from particle itself
    bool   print_angle            = true;
    float* template_psi           = new float[total_correlation_positions_sampled_view];
    float* template_theta         = new float[total_correlation_positions_sampled_view];
    float* template_phi           = new float[total_correlation_positions_sampled_view];
    float* particle_aligned_psi   = new float[total_correlation_positions_sampled_view];
    float* particle_aligned_theta = new float[total_correlation_positions_sampled_view];
    float* particle_aligned_phi   = new float[total_correlation_positions_sampled_view];

    // generate templates
    //#pragma omp parallel for schedule(dynamic, 1) default(shared) num_threads(max_threads) firstprivate(current_projection_image, current_projection_other, current_projection_correct_template)
    for ( current_search_position_tm = first_search_position; current_search_position_tm <= last_search_position_2; current_search_position_tm++ ) {
        for ( current_psi_tm = psi_start; current_psi_tm <= psi_max; current_psi_tm += psi_step_tm ) {
            angles_tm.Init(global_euler_search_tm.list_of_search_parameters[current_search_position_tm][0], global_euler_search_tm.list_of_search_parameters[current_search_position_tm][1], current_psi_tm, 0.0, 0.0);
            // generate projection from testing template for tm
            input_reconstruction_wrong.ExtractSlice(current_projection_other, angles_tm, 1.0f, false);
            //current_projection_2.SwapRealSpaceQuadrants( );
            current_projection_other.MultiplyPixelWise(projection_filter);
            current_projection_other.BackwardFFT( );
            current_projection_other.AddConstant(-current_projection_other.ReturnAverageOfRealValuesOnEdges( ));
            current_projection_other.AddConstant(-current_projection_other.ReturnAverageOfRealValues( ));
            variance = current_projection_other.ReturnSumOfSquares( ) - powf(current_projection_other.ReturnAverageOfRealValues( ), 2);
            current_projection_other.DivideByConstant(sqrtf(variance));
            //current_projection_2.AddGaussianNoise(10.0f);
            current_projection_other.ForwardFFT( );
            // Zeroing the central pixel is probably not doing anything useful...
            current_projection_other.ZeroCentralPixel( );

            // generate projection from particle's template for tm
            input_reconstruction_correct.ExtractSlice(current_projection_correct_template, angles_tm, 1.0f, false);
            //current_projection_2.SwapRealSpaceQuadrants( );
            current_projection_correct_template.MultiplyPixelWise(projection_filter);
            current_projection_correct_template.BackwardFFT( );
            current_projection_correct_template.AddConstant(-current_projection_correct_template.ReturnAverageOfRealValuesOnEdges( ));
            current_projection_correct_template.AddConstant(-current_projection_correct_template.ReturnAverageOfRealValues( ));
            variance = current_projection_correct_template.ReturnSumOfSquares( ) - powf(current_projection_correct_template.ReturnAverageOfRealValues( ), 2);
            current_projection_correct_template.DivideByConstant(sqrtf(variance));
            //current_projection_2.AddGaussianNoise(10.0f);
            current_projection_correct_template.ForwardFFT( );
            // Zeroing the central pixel is probably not doing anything useful...
            current_projection_correct_template.ZeroCentralPixel( );

            int view_counter = 0;
            for ( current_search_position_sampled_view = first_search_position; current_search_position_sampled_view <= last_search_position; current_search_position_sampled_view++ ) {
                for ( current_psi_sampled_view = psi_start; current_psi_sampled_view <= psi_max; current_psi_sampled_view += psi_step_sampled_view ) {

                    angles_sampled_view.Init(global_euler_search_sampled_view.list_of_search_parameters[current_search_position_sampled_view][0], global_euler_search_sampled_view.list_of_search_parameters[current_search_position_sampled_view][1], current_psi_sampled_view, 0.0, 0.0);
                    // generate particle from sampled view
                    input_reconstruction_particle.ExtractSlice(current_projection_image, angles_sampled_view, 1.0f, false);
                    current_projection_image.MultiplyPixelWise(projection_filter);
                    current_projection_image.BackwardFFT( );
                    current_projection_image.AddConstant(-current_projection_image.ReturnAverageOfRealValuesOnEdges( ));
                    current_projection_image.AddConstant(-current_projection_image.ReturnAverageOfRealValues( ));
                    variance = current_projection_image.ReturnSumOfSquares( ) - powf(current_projection_image.ReturnAverageOfRealValues( ), 2);
                    current_projection_image.DivideByConstant(sqrtf(variance));
                    current_projection_image.ForwardFFT( );

// calculate cc and ac
#ifdef MKL
                    // Use the MKL
                    vmcMulByConj(current_projection_image.real_memory_allocated / 2, reinterpret_cast<MKL_Complex8*>(current_projection_other.complex_values), reinterpret_cast<MKL_Complex8*>(current_projection_image.complex_values), reinterpret_cast<MKL_Complex8*>(current_projection_image.complex_values), VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
#else
                    for ( pixel_counter = 0; pixel_counter < current_projection_1.real_memory_allocated / 2; pixel_counter++ ) {
                        current_projection_1.complex_values[pixel_counter] = conj(current_projection_1.complex_values[pixel_counter]) * current_projection_2.complex_values[pixel_counter];
                    }
#endif
                    current_projection_image.SwapRealSpaceQuadrants( );
                    current_projection_image.BackwardFFT( );
                    cc_val = current_projection_image.ReturnCentralPixelValue( );

                    input_reconstruction_particle.ExtractSlice(current_projection_image, angles_sampled_view, 1.0f, false);
                    current_projection_image.MultiplyPixelWise(projection_filter);
                    current_projection_image.BackwardFFT( );
                    current_projection_image.AddConstant(-current_projection_image.ReturnAverageOfRealValuesOnEdges( ));
                    current_projection_image.AddConstant(-current_projection_image.ReturnAverageOfRealValues( ));
                    variance = current_projection_image.ReturnSumOfSquares( ) - powf(current_projection_image.ReturnAverageOfRealValues( ), 2);
                    current_projection_image.DivideByConstant(sqrtf(variance));
                    current_projection_image.ForwardFFT( );
#ifdef MKL
                    // Use the MKL
                    vmcMulByConj(current_projection_image.real_memory_allocated / 2, reinterpret_cast<MKL_Complex8*>(current_projection_correct_template.complex_values), reinterpret_cast<MKL_Complex8*>(current_projection_image.complex_values), reinterpret_cast<MKL_Complex8*>(current_projection_image.complex_values), VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
#else
                    for ( pixel_counter = 0; pixel_counter < current_projection_1.real_memory_allocated / 2; pixel_counter++ ) {
                        current_projection_1.complex_values[pixel_counter] = conj(current_projection_1.complex_values[pixel_counter]) * current_projection_2.complex_values[pixel_counter];
                    }
#endif
                    current_projection_image.SwapRealSpaceQuadrants( );
                    current_projection_image.BackwardFFT( );
                    ac_val = current_projection_image.ReturnCentralPixelValue( );

                    // find alignment using ac
                    if ( current_psi_sampled_view == current_psi_tm && global_euler_search_sampled_view.list_of_search_parameters[current_search_position_sampled_view][0] == global_euler_search_tm.list_of_search_parameters[current_search_position_tm][0] && global_euler_search_sampled_view.list_of_search_parameters[current_search_position_sampled_view][1] == global_euler_search_tm.list_of_search_parameters[current_search_position_tm][1] ) {
                        wxPrintf("view %i ac/cc = %f %f \n", view_counter, ac_val, cc_val);
                        //    continue;
                    }
                    // use ac to find aligned view

                    if ( ac_val > collected_ac_data[view_counter] ) {
                        collected_ac_data[view_counter]            = ac_val;
                        collected_cc_data[view_counter]            = cc_val;
                        collected_aligned_ratio_data[view_counter] = ac_val / cc_val;
                        particle_aligned_psi[view_counter]         = current_psi_sampled_view;
                        particle_aligned_theta[view_counter]       = global_euler_search_sampled_view.list_of_search_parameters[current_search_position_sampled_view][0];
                        particle_aligned_phi[view_counter]         = global_euler_search_sampled_view.list_of_search_parameters[current_search_position_sampled_view][1];
                        template_psi[view_counter]                 = current_psi_tm;
                        template_theta[view_counter]               = global_euler_search_tm.list_of_search_parameters[current_search_position_tm][0];
                        template_phi[view_counter]                 = global_euler_search_tm.list_of_search_parameters[current_search_position_tm][1];

                        //if ( print_angle )
                        //    wxPrintf("template view %f/%f/%f  image view %f/%f/%f  ac = %f  cc = %f\n", current_psi_tm, global_euler_search_tm.list_of_search_parameters[current_search_position_tm][0], global_euler_search_tm.list_of_search_parameters[current_search_position_tm][1], current_psi_sampled_view, global_euler_search_sampled_view.list_of_search_parameters[current_search_position_sampled_view][0], global_euler_search_sampled_view.list_of_search_parameters[current_search_position_sampled_view][1], ac_val, cc_val);
                    }
                    collected_ac_sum_data[view_counter] += (double)ac_val;
                    collected_cc_sum_data[view_counter] += (double)cc_val;
                    collected_ac_sum_of_squares_data[view_counter] += (double)powf(ac_val, 2);
                    collected_cc_sum_of_squares_data[view_counter] += (double)powf(cc_val, 2);
                    view_counter += 1;
                }
            }
        }
    }

    //#pragma omp parallel for num_threads(max_threads) schedule(static) private(current_psi_sampled_view, current_search_position_tm, current_psi_tm, angles_tm, angles_sampled_view, variance, ac_val, cc_val) firstprivate(current_projection_image, current_projection_other, current_projection_correct_template) reduction(+ \
                                                                                                                                                                                                                                                                                                                           : view_counter)

    /*
    for ( current_search_position_sampled_view = first_search_position; current_search_position_sampled_view <= last_search_position; current_search_position_sampled_view++ ) {
        for ( current_psi_sampled_view = psi_start; current_psi_sampled_view <= psi_max; current_psi_sampled_view += psi_step_sampled_view ) {
            angles_sampled_view.Init(global_euler_search_sampled_view.list_of_search_parameters[current_search_position_sampled_view][0], global_euler_search_sampled_view.list_of_search_parameters[current_search_position_sampled_view][1], current_psi_sampled_view, 0.0, 0.0);
            wxPrintf("worker %i starts @ rotation %i psi %f view %i\n", ReturnThreadNumberOfCurrentThread( ), current_search_position_sampled_view, current_psi_sampled_view, view_counter);
            // generate particle from sampled view
            input_reconstruction_particle.ExtractSlice(current_projection_image, angles_sampled_view, 1.0f, false);
            current_projection_image.MultiplyPixelWise(projection_filter);
            current_projection_image.BackwardFFT( );
            current_projection_image.AddConstant(-current_projection_image.ReturnAverageOfRealValuesOnEdges( ));
            current_projection_image.AddConstant(-current_projection_image.ReturnAverageOfRealValues( ));
            variance = current_projection_image.ReturnSumOfSquares( ) - powf(current_projection_image.ReturnAverageOfRealValues( ), 2);
            current_projection_image.DivideByConstant(sqrtf(variance));
            current_projection_image.ForwardFFT( );

            // generate templates
            for ( current_search_position_tm = first_search_position; current_search_position_tm <= last_search_position_2; current_search_position_tm++ ) {
                for ( current_psi_tm = psi_start; current_psi_tm <= psi_max; current_psi_tm += psi_step_tm ) {
                    angles_tm.Init(global_euler_search_tm.list_of_search_parameters[current_search_position_tm][0], global_euler_search_tm.list_of_search_parameters[current_search_position_tm][1], current_psi_tm, 0.0, 0.0);
                    // generate projection from testing template for tm
                    input_reconstruction_wrong.ExtractSlice(current_projection_other, angles_tm, 1.0f, false);
                    //current_projection_2.SwapRealSpaceQuadrants( );
                    current_projection_other.MultiplyPixelWise(projection_filter);
                    current_projection_other.BackwardFFT( );
                    current_projection_other.AddConstant(-current_projection_other.ReturnAverageOfRealValuesOnEdges( ));
                    current_projection_other.AddConstant(-current_projection_other.ReturnAverageOfRealValues( ));
                    variance = current_projection_other.ReturnSumOfSquares( ) - powf(current_projection_other.ReturnAverageOfRealValues( ), 2);
                    current_projection_other.DivideByConstant(sqrtf(variance));
                    //current_projection_2.AddGaussianNoise(10.0f);
                    current_projection_other.ForwardFFT( );
                    // Zeroing the central pixel is probably not doing anything useful...
                    current_projection_other.ZeroCentralPixel( );

                    // generate projection from particle's template for tm
                    input_reconstruction_correct.ExtractSlice(current_projection_correct_template, angles_tm, 1.0f, false);
                    //current_projection_2.SwapRealSpaceQuadrants( );
                    current_projection_correct_template.MultiplyPixelWise(projection_filter);
                    current_projection_correct_template.BackwardFFT( );
                    current_projection_correct_template.AddConstant(-current_projection_correct_template.ReturnAverageOfRealValuesOnEdges( ));
                    current_projection_correct_template.AddConstant(-current_projection_correct_template.ReturnAverageOfRealValues( ));
                    variance = current_projection_correct_template.ReturnSumOfSquares( ) - powf(current_projection_correct_template.ReturnAverageOfRealValues( ), 2);
                    current_projection_correct_template.DivideByConstant(sqrtf(variance));
                    //current_projection_2.AddGaussianNoise(10.0f);
                    current_projection_correct_template.ForwardFFT( );
                    // Zeroing the central pixel is probably not doing anything useful...
                    current_projection_correct_template.ZeroCentralPixel( );

// calculate cc and ac
#ifdef MKL
                    // Use the MKL
                    vmcMulByConj(current_projection_other.real_memory_allocated / 2, reinterpret_cast<MKL_Complex8*>(current_projection_image.complex_values), reinterpret_cast<MKL_Complex8*>(current_projection_other.complex_values), reinterpret_cast<MKL_Complex8*>(current_projection_other.complex_values), VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
#else
                    for ( pixel_counter = 0; pixel_counter < current_projection_1.real_memory_allocated / 2; pixel_counter++ ) {
                        current_projection_1.complex_values[pixel_counter] = conj(current_projection_1.complex_values[pixel_counter]) * current_projection_2.complex_values[pixel_counter];
                    }
#endif
                    current_projection_other.SwapRealSpaceQuadrants( );
                    current_projection_other.BackwardFFT( );
                    cc_val = current_projection_other.ReturnCentralPixelValue( );

#ifdef MKL
                    // Use the MKL
                    vmcMulByConj(current_projection_correct_template.real_memory_allocated / 2, reinterpret_cast<MKL_Complex8*>(current_projection_image.complex_values), reinterpret_cast<MKL_Complex8*>(current_projection_correct_template.complex_values), reinterpret_cast<MKL_Complex8*>(current_projection_correct_template.complex_values), VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
#else
                    for ( pixel_counter = 0; pixel_counter < current_projection_1.real_memory_allocated / 2; pixel_counter++ ) {
                        current_projection_1.complex_values[pixel_counter] = conj(current_projection_1.complex_values[pixel_counter]) * current_projection_2.complex_values[pixel_counter];
                    }
#endif
                    current_projection_correct_template.SwapRealSpaceQuadrants( );
                    current_projection_correct_template.BackwardFFT( );
                    ac_val = current_projection_correct_template.ReturnCentralPixelValue( );

                    // find alignment using ac
                    if ( current_psi_sampled_view == current_psi_tm && global_euler_search_sampled_view.list_of_search_parameters[current_search_position_sampled_view][0] == global_euler_search_tm.list_of_search_parameters[current_search_position_tm][0] && global_euler_search_sampled_view.list_of_search_parameters[current_search_position_sampled_view][1] == global_euler_search_tm.list_of_search_parameters[current_search_position_tm][1] ) {
                        wxPrintf("worker %i reports view %i ac/cc = %f %f \n", ReturnThreadNumberOfCurrentThread( ), view_counter, ac_val, cc_val);
                        //    continue;
                    }

                    // use ac to find aligned view
                    if ( ac_val > collected_ac_data[view_counter] ) {
                        collected_ac_data[view_counter]            = ac_val;
                        collected_cc_data[view_counter]            = cc_val;
                        collected_aligned_ratio_data[view_counter] = ac_val / cc_val;
                        particle_aligned_psi[view_counter]         = current_psi_sampled_view;
                        particle_aligned_theta[view_counter]       = global_euler_search_sampled_view.list_of_search_parameters[current_search_position_sampled_view][0];
                        particle_aligned_phi[view_counter]         = global_euler_search_sampled_view.list_of_search_parameters[current_search_position_sampled_view][1];
                        template_psi[view_counter]                 = current_psi_tm;
                        template_theta[view_counter]               = global_euler_search_tm.list_of_search_parameters[current_search_position_tm][0];
                        template_phi[view_counter]                 = global_euler_search_tm.list_of_search_parameters[current_search_position_tm][1];

                        //if ( print_angle )
                        //   wxPrintf("template view %f/%f/%f  image view %f/%f/%f  ac = %f  cc = %f\n", current_psi_tm, global_euler_search_tm.list_of_search_parameters[current_search_position_tm][0], global_euler_search_tm.list_of_search_parameters[current_search_position_tm][1], current_psi_sampled_view, global_euler_search_sampled_view.list_of_search_parameters[current_search_position_sampled_view][0], global_euler_search_sampled_view.list_of_search_parameter[current_search_position_sampled_view][1], ac_val, cc_val);
                    }
                    //wxPrintf("ac = %f  cc = %f   ratio = %f\n", cc_val, ac_val, ac_val / cc_val);
                    collected_ac_sum_data[view_counter] += (double)ac_val;
                    collected_cc_sum_data[view_counter] += (double)cc_val;
                    collected_ac_sum_of_squares_data[view_counter] += (double)powf(ac_val, 2);
                    collected_cc_sum_of_squares_data[view_counter] += (double)powf(cc_val, 2);

                } // end of tm psi
            } // end of tm
            view_counter = view_counter + 1;
        }
    }
*/
    //output_file.Close( );
    for ( int line_counter = 0; line_counter < total_correlation_positions_sampled_view; line_counter++ ) {
        collected_ac_sum_data[line_counter] /= total_correlation_positions_tm; // avg of ac from all tm projections
        collected_cc_sum_data[line_counter] /= total_correlation_positions_tm;
        collected_ac_sum_of_squares_data[line_counter] /= total_correlation_positions_tm; // avg of ac^2 from all tm projections
        collected_cc_sum_of_squares_data[line_counter] /= total_correlation_positions_tm;
        //wxPrintf("line %i ac = %lf ratio = %lf avg = %lf var = %lf\n", line_counter, collected_ac_data[line_counter], collected_aligned_ratio_data[line_counter], collected_avg_data[line_counter], collected_var_data[line_counter]);
    }

    // now we have M maximums, M avgs, M sum of squares, generate statistics across all M views
    float snr_multi_views_sum_unscaled = 0;
    float snr_multi_views_sum_scaled   = 0;
    float var_of_ccs, var_of_acs;
    float snr_multi_views_sum_of_squares_scaled   = 0;
    float snr_multi_views_sum_of_squares_unscaled = 0;
    float scaled_ac_single_view, scaled_cc_single_view, scaled_snr_single_view;

    for ( int line_counter = 0; line_counter < total_correlation_positions_sampled_view; line_counter++ ) {
        wxPrintf("particle view %i @ %f %f %f aligned with template @ %f %f %f = %lf %lf %lf\n", line_counter, particle_aligned_psi[line_counter], particle_aligned_theta[line_counter], particle_aligned_phi[line_counter], template_psi[line_counter], template_theta[line_counter], template_phi[line_counter], collected_ac_data[line_counter], collected_cc_data[line_counter], collected_aligned_ratio_data[line_counter]);
        snr_multi_views_sum_unscaled += collected_aligned_ratio_data[line_counter];
        snr_multi_views_sum_of_squares_unscaled += powf(collected_aligned_ratio_data[line_counter], 2);
        var_of_ccs             = collected_cc_sum_of_squares_data[line_counter] - powf(collected_cc_sum_data[line_counter], 2);
        var_of_acs             = collected_ac_sum_of_squares_data[line_counter] - powf(collected_ac_sum_data[line_counter], 2);
        scaled_ac_single_view  = (collected_ac_data[line_counter] - collected_ac_sum_data[line_counter]) / sqrtf(var_of_acs);
        scaled_cc_single_view  = (collected_cc_data[line_counter] - collected_cc_sum_data[line_counter]) / sqrtf(var_of_ccs);
        scaled_snr_single_view = scaled_ac_single_view / scaled_cc_single_view;
        snr_multi_views_sum_scaled += scaled_snr_single_view;
        snr_multi_views_sum_of_squares_scaled += powf(scaled_snr_single_view, 2);
        wxPrintf("view %i AC mean = %lf CC mean = %lf\n", line_counter, collected_ac_sum_data[line_counter], collected_cc_sum_data[line_counter]);
    }

    wxPrintf("Avg of multiview aligned AC/CC %f\n", snr_multi_views_sum_unscaled / total_correlation_positions_sampled_view);
    float var_unscaled = snr_multi_views_sum_of_squares_unscaled / total_correlation_positions_sampled_view - powf(snr_multi_views_sum_unscaled / total_correlation_positions_sampled_view, 2);
    wxPrintf("SD of multiview aligned AC/CC %f\n", sqrtf(var_unscaled));
    wxPrintf("Avg of multiview aligned AC/CC (scaled) %f\n", snr_multi_views_sum_scaled / total_correlation_positions_sampled_view);
    float var_scaled = snr_multi_views_sum_of_squares_scaled / total_correlation_positions_sampled_view - powf(snr_multi_views_sum_scaled / total_correlation_positions_sampled_view, 2);
    wxPrintf("SD of multiview aligned AC/CC (scaled) %f\n", sqrtf(var_scaled));

    delete[] collected_ac_data;
    delete[] collected_cc_data;
    delete[] collected_aligned_ratio_data;
    delete[] collected_ac_sum_data;
    delete[] collected_cc_sum_data;
    delete[] collected_ac_sum_of_squares_data;
    delete[] collected_cc_sum_of_squares_data;

    return true;
}

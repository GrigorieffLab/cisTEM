#include "../../core/core_headers.h"
#include "../../core/cistem_constants.h"

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
    wxString log_output_file;
    wxString scaled_mip_ac_filename, scaled_mip_cc_filename, mip_ac_filename, mip_cc_filename, avg_ac_filename, avg_cc_filename, std_ac_filename, std_cc_filename, data_directory_name;

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
    bool     use_gpu_input                  = false;

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
    log_output_file                        = my_input->GetFilenameFromUser("Log file for recording meta data", "Log output file", "log.txt", false);
#ifdef ENABLEGPU
    use_gpu_input = my_input->GetYesNoFromUser("Use GPU", "Offload expensive calcs to GPU", "No");
    max_threads   = my_input->GetIntFromUser("Max. threads to use for calculation", "when threading, what is the max threads to run", "1", 1);
#endif
    scaled_mip_ac_filename = my_input->GetFilenameFromUser("Output scaled mip image ac", "Output scaled mip image ac", "scaled_mip_ac.mrc", false);
    scaled_mip_cc_filename = my_input->GetFilenameFromUser("Output scaled mip image cc", "Output scaled mip image ac", "scaled_mip_cc.mrc", false);
    mip_ac_filename        = my_input->GetFilenameFromUser("Output mip image ac", "Output mip image ac", "mip_ac.mrc", false);
    mip_cc_filename        = my_input->GetFilenameFromUser("Output mip image cc", "Output mip image cc", "mip_cc.mrc", false);
    avg_ac_filename        = my_input->GetFilenameFromUser("Output avg image ac", "Output avg image ac", "avg_ac.mrc", false);
    avg_cc_filename        = my_input->GetFilenameFromUser("Output avg image cc", "Output avg image cc", "avg_cc.mrc", false);
    std_ac_filename        = my_input->GetFilenameFromUser("Output std image ac", "Output std image ac", "std_cc.mrc", false);
    std_cc_filename        = my_input->GetFilenameFromUser("Output std image cc", "Output std image cc", "std_cc.mrc", false);
    data_directory_name    = my_input->GetFilenameFromUser("Name for data directory", "path to data directory", "60_120_5_2.5", false);

    int first_search_position             = -1;
    int last_search_position_sampled_view = -1;
    int last_search_position_tm           = -1;

    delete my_input;

    my_current_job.ManualSetArguments("ttttfffffffffifftfiiffitbittttttttt",
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
                                      last_search_position_sampled_view,
                                      angular_step_tm,
                                      in_plane_angular_step_tm,
                                      last_search_position_tm,
                                      log_output_file.ToUTF8( ).data( ),
                                      use_gpu_input,
                                      max_threads,
                                      scaled_mip_ac_filename.ToUTF8( ).data( ),
                                      scaled_mip_cc_filename.ToUTF8( ).data( ),
                                      mip_ac_filename.ToUTF8( ).data( ),
                                      mip_cc_filename.ToUTF8( ).data( ),
                                      avg_ac_filename.ToUTF8( ).data( ),
                                      avg_cc_filename.ToUTF8( ).data( ),
                                      std_ac_filename.ToUTF8( ).data( ),
                                      std_cc_filename.ToUTF8( ).data( ),
                                      data_directory_name.ToUTF8( ).data( ));
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
    float    high_resolution_limit_search      = my_current_job.arguments[11].ReturnFloatArgument( );
    float    angular_step_sampling             = my_current_job.arguments[12].ReturnFloatArgument( );
    int      best_parameters_to_keep           = my_current_job.arguments[13].ReturnIntegerArgument( );
    float    padding                           = my_current_job.arguments[14].ReturnFloatArgument( );
    float    phase_shift                       = my_current_job.arguments[15].ReturnFloatArgument( );
    wxString my_symmetry                       = my_current_job.arguments[16].ReturnStringArgument( );
    float    in_plane_angular_step_sampling    = my_current_job.arguments[17].ReturnFloatArgument( );
    int      first_search_position             = my_current_job.arguments[18].ReturnIntegerArgument( );
    int      last_search_position_sampled_view = my_current_job.arguments[19].ReturnIntegerArgument( );
    float    angular_step_tm                   = my_current_job.arguments[20].ReturnFloatArgument( );
    float    in_plane_angular_step_tm          = my_current_job.arguments[21].ReturnFloatArgument( );
    int      last_search_position_tm           = my_current_job.arguments[22].ReturnIntegerArgument( );
    wxString log_output_file                   = my_current_job.arguments[23].ReturnStringArgument( );
    bool     use_gpu                           = my_current_job.arguments[24].ReturnBoolArgument( );
    int      max_threads                       = my_current_job.arguments[25].ReturnIntegerArgument( );
    wxString scaled_mip_ac_filename            = my_current_job.arguments[26].ReturnStringArgument( );
    wxString scaled_mip_cc_filename            = my_current_job.arguments[27].ReturnStringArgument( );
    wxString mip_ac_filename                   = my_current_job.arguments[28].ReturnStringArgument( );
    wxString mip_cc_filename                   = my_current_job.arguments[29].ReturnStringArgument( );
    wxString avg_ac_filename                   = my_current_job.arguments[30].ReturnStringArgument( );
    wxString avg_cc_filename                   = my_current_job.arguments[31].ReturnStringArgument( );
    wxString std_ac_filename                   = my_current_job.arguments[32].ReturnStringArgument( );
    wxString std_cc_filename                   = my_current_job.arguments[33].ReturnStringArgument( );
    wxString data_directory_name               = my_current_job.arguments[34].ReturnStringArgument( );

    // This condition applies to GUI and CLI - it is just a recommendation to the user.
    if ( use_gpu && max_threads <= 1 ) {
        SendInfo("Warning, you are only using one thread on the GPU. Suggested minimum is 2. Check compute saturation using nvidia-smi -l 1\n");
    }
    if ( ! use_gpu ) {
        SendInfo("GPU disabled\nCan use up to 44 threads on roma\n.");
    }

    int  padded_dimensions_x;
    int  padded_dimensions_y;
    int  pad_factor                       = 6;
    int  number_of_rotations_sampled_view = 0;
    int  number_of_rotations_tm           = 0;
    long total_correlation_positions_sampled_view, total_correlation_positions_tm;
    long current_correlation_position_sampled_view, current_correlation_position_tm;
    long total_correlation_positions_sampled_view_per_thread;
    long pixel_counter;

    int   current_search_position_sampled_view, current_search_position_tm;
    float current_psi_sampled_view, current_psi_tm;
    float psi_step_sampled_view = in_plane_angular_step_sampling;
    float psi_step_tm           = in_plane_angular_step_tm;
    float psi_max               = 360.0f;
    float psi_start             = 0.0f;

    ParameterMap parameter_map; // needed for euler search init
    parameter_map.SetAllTrue( );
    float variance;
    float ccs_one_view_sum            = 0.0;
    float ccs_one_view_sum_of_squares = 0.0;
    float ccs_one_view_var, ccs_one_view_max, ccs_one_view_max_scaled;
    float ccs_multi_views_sum                   = 0.0;
    float ccs_multi_views_sum_scaled            = 0.0;
    float ccs_multi_views_sum_of_squares        = 0.0;
    float ccs_multi_views_sum_of_squares_scaled = 0.0;

    Curve           whitening_filter;
    Curve           number_of_terms;
    NumericTextFile log_file(log_output_file, OPEN_TO_WRITE, 1);

    Image           input_reconstruction_particle, input_reconstruction_correct, input_reconstruction_wrong;
    ImageFile       input_search_image_file;
    ImageFile       input_reconstruction_particle_file, input_reconstruction_correct_template_file, input_reconstruction_wrong_template_file;
    Image           input_image;
    EulerSearch     global_euler_search_sampled_view, global_euler_search_tm;
    AnglesAndShifts angles_sampled_view, angles_tm;

    input_search_image_file.OpenFile(input_search_images_filename.ToStdString( ), false);
    input_image.ReadSlice(&input_search_image_file, 1);
    input_reconstruction_particle_file.OpenFile(input_reconstruction_particle_filename.ToStdString( ), false);
    input_reconstruction_correct_template_file.OpenFile(input_reconstruction_correct_filename.ToStdString( ), false);
    input_reconstruction_wrong_template_file.OpenFile(input_reconstruction_wrong_filename.ToStdString( ), false);
    input_reconstruction_particle.ReadSlices(&input_reconstruction_particle_file, 1, input_reconstruction_particle_file.ReturnNumberOfSlices( )); // particle in image
    input_reconstruction_correct.ReadSlices(&input_reconstruction_correct_template_file, 1, input_reconstruction_correct_template_file.ReturnNumberOfSlices( )); // correct template
    input_reconstruction_wrong.ReadSlices(&input_reconstruction_wrong_template_file, 1, input_reconstruction_wrong_template_file.ReturnNumberOfSlices( )); // wrong template

    // 1 is coarse search; 2 is finer TM search
    // TODO 1. normalize (done) 2. pad image with mean and pad template with 0 to same size 3. pad template to remove aliasing (done)
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

    first_search_position             = 0;
    last_search_position_sampled_view = global_euler_search_sampled_view.number_of_search_positions - 1;
    last_search_position_tm           = global_euler_search_tm.number_of_search_positions - 1;

    total_correlation_positions_sampled_view  = 0;
    total_correlation_positions_tm            = 0;
    current_correlation_position_sampled_view = 0;
    current_correlation_position_tm           = 0;

    for ( current_search_position_sampled_view = first_search_position; current_search_position_sampled_view <= last_search_position_sampled_view; current_search_position_sampled_view++ ) {
        //loop over each rotation angle

        for ( current_psi_sampled_view = psi_start; current_psi_sampled_view <= psi_max; current_psi_sampled_view += psi_step_sampled_view ) {
            total_correlation_positions_sampled_view++;
        }
    }

    for ( current_psi_sampled_view = psi_start; current_psi_sampled_view <= psi_max; current_psi_sampled_view += psi_step_sampled_view ) {
        number_of_rotations_sampled_view++;
    }

    for ( current_search_position_tm = first_search_position; current_search_position_tm <= last_search_position_tm; current_search_position_tm++ ) {
        //loop over each rotation angle

        for ( current_psi_tm = psi_start; current_psi_tm <= psi_max; current_psi_tm += psi_step_tm ) {
            total_correlation_positions_tm++;
        }
    }

    for ( current_psi_tm = psi_start; current_psi_tm <= psi_max; current_psi_tm += psi_step_tm ) {
        number_of_rotations_tm++;
    }

    wxPrintf("Outside Loop - sampled views in an image:\n");
    wxPrintf("Searching %i positions on the Euler sphere (first-last: %i-%i)\n", last_search_position_sampled_view - first_search_position, first_search_position, last_search_position_sampled_view);
    wxPrintf("Searching %i rotations per position.\n", number_of_rotations_sampled_view);
    wxPrintf("There are %li correlation positions total.\n\n", total_correlation_positions_sampled_view);

    wxPrintf("Inside Loop - tm rotations:\n");
    wxPrintf("Searching %i positions on the Euler sphere (first-last: %i-%i)\n", last_search_position_tm - first_search_position, first_search_position, last_search_position_tm);
    wxPrintf("Searching %i rotations per position.\n", number_of_rotations_tm);
    wxPrintf("There are %li correlation positions total.\n\n", total_correlation_positions_tm);

    // These vars are only needed in the GPU code, but also need to be set out here to compile.
    bool first_gpu_loop = true;
    int  nGPUs          = 1;
    int  nJobs          = last_search_position_sampled_view - first_search_position + 1;
    if ( use_gpu && max_threads > nJobs ) {
        wxPrintf("\n\tWarning, you request more threads (%d) than there are search positions (%d)\n", max_threads, nJobs);
        max_threads = nJobs;
    }

    int minPos = first_search_position;
    int maxPos = last_search_position_sampled_view;
    int incPos = (nJobs) / (max_threads);

#ifdef ENABLEGPU
    TemplateSnrRatioCore* GPU;
    DeviceManager         gpuDev;
#endif

    if ( use_gpu ) {
        total_correlation_positions_sampled_view_per_thread = total_correlation_positions_sampled_view / max_threads;

#ifdef ENABLEGPU
        //    checkCudaErrors(cudaGetDeviceCount(&nGPUs));
        GPU = new TemplateSnrRatioCore[max_threads];
        gpuDev.Init(nGPUs);

#endif
    }

    ProgressBar* my_progress;
    my_progress = new ProgressBar(total_correlation_positions_sampled_view_per_thread);

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

    //whitening_filter.WriteToFile("/tmp/filter.txt");
    //input_image.ApplyCurveFilter(&whitening_filter);
    //input_image.ZeroCentralPixel( );
    //input_image.DivideByConstant(sqrtf(input_image.ReturnSumOfSquares( )));

    input_reconstruction_particle.ForwardFFT( );
    input_reconstruction_particle.ZeroCentralPixel( );
    input_reconstruction_particle.SwapRealSpaceQuadrants( );

    // only pad templates to remove aliasing
    if ( padding != 1.0f ) {
        input_reconstruction_correct.Resize(input_reconstruction_correct.logical_x_dimension * padding, input_reconstruction_correct.logical_y_dimension * padding, input_reconstruction_correct.logical_z_dimension * padding, input_reconstruction_correct.ReturnAverageOfRealValuesOnEdges( ));
    }
    input_reconstruction_correct.ForwardFFT( );
    input_reconstruction_correct.ZeroCentralPixel( );
    input_reconstruction_correct.SwapRealSpaceQuadrants( );

    if ( padding != 1.0f ) {
        input_reconstruction_wrong.Resize(input_reconstruction_wrong.logical_x_dimension * padding, input_reconstruction_wrong.logical_y_dimension * padding, input_reconstruction_wrong.logical_z_dimension * padding, input_reconstruction_wrong.ReturnAverageOfRealValuesOnEdges( ));
    }
    input_reconstruction_wrong.ForwardFFT( );
    input_reconstruction_wrong.ZeroCentralPixel( );
    input_reconstruction_wrong.SwapRealSpaceQuadrants( );

    CTF   input_ctf;
    Image projection_filter;
    projection_filter.Allocate(input_reconstruction_particle.logical_x_dimension, input_reconstruction_particle.logical_y_dimension, false);
    input_ctf.Init(voltage_kV, spherical_aberration_mm, amplitude_contrast, defocus1, defocus2, defocus_angle, 0.0, 0.0, 0.0, pixel_size, deg_2_rad(phase_shift));
    input_ctf.SetDefocus(defocus1 / pixel_size, defocus2 / pixel_size, deg_2_rad(defocus_angle));
    projection_filter.CalculateCTFImage(input_ctf);
    projection_filter.ApplyCurveFilter(&whitening_filter);

    // images for storing mip
    Image max_intensity_projection_ac, max_intensity_projection_cc;
    max_intensity_projection_ac.Allocate(input_reconstruction_particle.logical_x_dimension, input_reconstruction_particle.logical_y_dimension, int(total_correlation_positions_sampled_view));
    max_intensity_projection_ac.SetToConstant(-FLT_MAX);
    max_intensity_projection_cc.Allocate(input_reconstruction_particle.logical_x_dimension, input_reconstruction_particle.logical_y_dimension, int(total_correlation_positions_sampled_view));
    max_intensity_projection_cc.SetToConstant(-FLT_MAX);
    Image correlation_pixel_sum_ac, correlation_pixel_sum_of_squares_ac, correlation_pixel_sum_cc, correlation_pixel_sum_of_squares_cc;
    correlation_pixel_sum_ac.Allocate(input_reconstruction_particle.logical_x_dimension, input_reconstruction_particle.logical_y_dimension, int(total_correlation_positions_sampled_view));
    correlation_pixel_sum_of_squares_ac.Allocate(input_reconstruction_particle.logical_x_dimension, input_reconstruction_particle.logical_y_dimension, int(total_correlation_positions_sampled_view));
    correlation_pixel_sum_cc.Allocate(input_reconstruction_particle.logical_x_dimension, input_reconstruction_particle.logical_y_dimension, int(total_correlation_positions_sampled_view));
    correlation_pixel_sum_of_squares_cc.Allocate(input_reconstruction_particle.logical_x_dimension, input_reconstruction_particle.logical_y_dimension, int(total_correlation_positions_sampled_view));
    correlation_pixel_sum_ac.SetToConstant(0.0f);
    correlation_pixel_sum_cc.SetToConstant(0.0f);
    correlation_pixel_sum_of_squares_ac.SetToConstant(0.0f);
    correlation_pixel_sum_of_squares_cc.SetToConstant(0.0f);

    // step one: generate N projections from template 2 that is different from particle itself
    bool   print_angle            = true;
    float* template_psi           = new float[total_correlation_positions_sampled_view];
    float* template_theta         = new float[total_correlation_positions_sampled_view];
    float* template_phi           = new float[total_correlation_positions_sampled_view];
    float* particle_aligned_psi   = new float[total_correlation_positions_sampled_view];
    float* particle_aligned_theta = new float[total_correlation_positions_sampled_view];
    float* particle_aligned_phi   = new float[total_correlation_positions_sampled_view];
    int    j, view_counter;
    int    current_x, current_y;
    Image  current_projection_image, current_projection_other, current_projection_correct_template, padded_projection;
    float  ac_val, ac_max, cc_val, cc_max;

    float img_avg = 0.0f;
    float img_std = 0.0f;

    current_projection_image.Allocate(input_reconstruction_particle.logical_x_dimension, input_reconstruction_particle.logical_y_dimension, 1, false);

    for ( current_search_position_sampled_view = first_search_position; current_search_position_sampled_view <= last_search_position_sampled_view; current_search_position_sampled_view++ ) {
        for ( j = 0; j < number_of_rotations_sampled_view; j++ ) {
            current_psi_sampled_view = psi_start + j * psi_step_sampled_view;
            view_counter             = current_search_position_sampled_view * number_of_rotations_sampled_view + j;
            //wxPrintf("worker #%i working on view #%i\n", ReturnThreadNumberOfCurrentThread( ), view_counter);
            angles_sampled_view.Init(global_euler_search_sampled_view.list_of_search_parameters[current_search_position_sampled_view][0], global_euler_search_sampled_view.list_of_search_parameters[current_search_position_sampled_view][1], current_psi_sampled_view, 0.0, 0.0);
            // generate particle from sampled view (in TM we applied projection filter first then normalized the templates)
            input_reconstruction_particle.ExtractSlice(current_projection_image, angles_sampled_view, 1.0f, false);
            //current_projection_image.SwapRealSpaceQuadrants( );
            current_projection_image.MultiplyPixelWise(projection_filter);
            current_projection_image.BackwardFFT( );
            //wxPrintf("view %i before scaling avg = %f std = %f\n", view_counter, current_projection_image.ReturnAverageOfRealValues( ), sqrtf(current_projection_image.ReturnVarianceOfRealValues( )));
            //current_projection_image.QuickAndDirtyWriteSlice(wxString::Format("view_%i_image.mrc", view_counter).ToStdString( ), 1);

            img_avg += current_projection_image.ReturnAverageOfRealValues( );
        }
    }
    img_avg /= total_correlation_positions_sampled_view;

    for ( current_search_position_sampled_view = first_search_position; current_search_position_sampled_view <= last_search_position_sampled_view; current_search_position_sampled_view++ ) {
        for ( j = 0; j < number_of_rotations_sampled_view; j++ ) {
            current_psi_sampled_view = psi_start + j * psi_step_sampled_view;
            view_counter             = current_search_position_sampled_view * number_of_rotations_sampled_view + j;
            //wxPrintf("worker #%i working on view #%i\n", ReturnThreadNumberOfCurrentThread( ), view_counter);
            angles_sampled_view.Init(global_euler_search_sampled_view.list_of_search_parameters[current_search_position_sampled_view][0], global_euler_search_sampled_view.list_of_search_parameters[current_search_position_sampled_view][1], current_psi_sampled_view, 0.0, 0.0);
            // generate particle from sampled view
            input_reconstruction_particle.ExtractSlice(current_projection_image, angles_sampled_view, 1.0f, false);
            //current_projection_image.SwapRealSpaceQuadrants( ); //

            current_projection_image.MultiplyPixelWise(projection_filter);
            current_projection_image.BackwardFFT( );
            current_projection_image.AddConstant(-img_avg);
            img_std += current_projection_image.ReturnSumOfSquares( ) * current_projection_image.number_of_real_space_pixels;
            //current_projection_image.QuickAndDirtyWriteSlice(wxString::Format("view_%i_image.mrc", view_counter).ToStdString( ), 1);
        }
    }
    img_std = sqrtf(img_std / current_projection_image.number_of_real_space_pixels / total_correlation_positions_sampled_view);
    wxPrintf("all images avg*1000 = %f std*1000 = %f\n", img_avg * 1000, img_std * 1000);
    log_file.WriteCommentLine("all images avg = %f std = %f\n", img_avg, img_std);

    // allocate current_projection before gpu code
    // use particle to determine size since template volumes may be padded
    current_projection_other.Allocate(input_reconstruction_particle.logical_x_dimension, input_reconstruction_particle.logical_y_dimension, 1, false);
    current_projection_correct_template.Allocate(input_reconstruction_particle.logical_x_dimension, input_reconstruction_particle.logical_y_dimension, 1, false);

    wxDateTime overall_start;
    wxDateTime overall_finish;
    overall_start                         = wxDateTime::Now( );
    float actual_number_of_ccs_calculated = 0.0;
    if ( use_gpu ) {
#ifdef ENABLEGPU
#pragma omp parallel num_threads(max_threads)
        {
            int tIDX = ReturnThreadNumberOfCurrentThread( );
            gpuDev.SetGpu(tIDX);

            if ( first_gpu_loop ) {

                int t_first_search_position = first_search_position + (tIDX * incPos);
                int t_last_search_position  = first_search_position + (incPos - 1) + (tIDX * incPos);

                if ( tIDX == (max_threads - 1) )
                    t_last_search_position = maxPos;

                GPU[tIDX].Init(this, input_reconstruction_particle, input_reconstruction_correct, input_reconstruction_wrong, input_image, current_projection_image, current_projection_correct_template, current_projection_other, psi_max, psi_start, psi_step_sampled_view, psi_step_tm, angles_sampled_view, angles_tm, global_euler_search_sampled_view, global_euler_search_tm, t_first_search_position, t_last_search_position, first_search_position, last_search_position_tm, my_progress, number_of_rotations_sampled_view, total_correlation_positions_sampled_view, total_correlation_positions_sampled_view_per_thread, img_avg, img_std, data_directory_name);

                wxPrintf("%d\n", tIDX);
                wxPrintf("%d\n", t_first_search_position);
                wxPrintf("%d\n", t_last_search_position);
                wxPrintf("Staring TemplateMatchingCore object %d to work on position range %d-%d\n", tIDX, t_first_search_position, t_last_search_position);

                first_gpu_loop = false;
            }
        }
#endif
    }

    // TODO: runinnerloop

    if ( use_gpu ) {
#ifdef ENABLEGPU
        //            wxPrintf("\n\n\t\tsizeI defI %d %d\n\n\n", size_i, defocus_i);

#pragma omp parallel num_threads(max_threads)
        {
            int tIDX = ReturnThreadNumberOfCurrentThread( );
            gpuDev.SetGpu(tIDX);

            GPU[tIDX].RunInnerLoop(projection_filter, tIDX, current_correlation_position_sampled_view);

#pragma omp critical
            {

                Image mip_buffer_ac, mip_buffer_cc; // FIXME is there better way to allocate memory for d_cc and d_sum? all views or partial views?
                mip_buffer_ac.CopyFrom(&max_intensity_projection_ac);

                mip_buffer_cc.CopyFrom(&max_intensity_projection_cc);

                // Image psi_buffer;
                // psi_buffer.CopyFrom(&max_intensity_projection);
                // Image phi_buffer;
                // phi_buffer.CopyFrom(&max_intensity_projection);
                // Image theta_buffer;
                // theta_buffer.CopyFrom(&max_intensity_projection);

                GPU[tIDX].d_max_intensity_projection_ac_all_views.CopyDeviceToHost(mip_buffer_ac, true, false);

                //  mip_buffer_ac.QuickAndDirtyWriteSlices(wxString::Format("84_5_2.5/tmpMipBuffer_ac_%i.mrc", ReturnThreadNumberOfCurrentThread( )).ToStdString( ), 1, mip_buffer_ac.logical_z_dimension);
                GPU[tIDX].d_max_intensity_projection_cc_all_views.CopyDeviceToHost(mip_buffer_cc, true, false);
                //  mip_buffer_cc.QuickAndDirtyWriteSlices(wxString::Format("84_5_2.5/tmpMipBuffer_cc_%i.mrc", ReturnThreadNumberOfCurrentThread( )).ToStdString( ), 1, mip_buffer_cc.logical_z_dimension);

                // GPU[tIDX].d_best_psi.CopyDeviceToHost(psi_buffer, true, false);
                // GPU[tIDX].d_best_phi.CopyDeviceToHost(phi_buffer, true, false);
                // GPU[tIDX].d_best_theta.CopyDeviceToHost(theta_buffer, true, false);

                //                    mip_buffer.QuickAndDirtyWriteSlice("/tmp/tmpMipBuffer.mrc",1,1);
                // TODO should prob aggregate these across all workers
                // TODO add a copySum method that allocates a pinned buffer, copies there then sumes into the wanted image.
                Image sum_ac, sum_cc;
                Image sumSq_ac, sumSq_cc;

                sum_ac.Allocate(input_reconstruction_particle.logical_x_dimension, input_reconstruction_particle.logical_y_dimension, int(total_correlation_positions_sampled_view));
                sumSq_ac.Allocate(input_reconstruction_particle.logical_x_dimension, input_reconstruction_particle.logical_y_dimension, int(total_correlation_positions_sampled_view));
                sum_cc.Allocate(input_reconstruction_particle.logical_x_dimension, input_reconstruction_particle.logical_y_dimension, int(total_correlation_positions_sampled_view));
                sumSq_cc.Allocate(input_reconstruction_particle.logical_x_dimension, input_reconstruction_particle.logical_y_dimension, int(total_correlation_positions_sampled_view));

                sum_ac.SetToConstant(0.0f);
                sumSq_ac.SetToConstant(0.0f);
                sum_cc.SetToConstant(0.0f);
                sumSq_cc.SetToConstant(0.0f);

                GPU[tIDX].d_sum_ac.CopyDeviceToHost(sum_ac, true, false);
                // sum_ac.QuickAndDirtyWriteSlices("cpu_sum_ac.mrc", 1, sum_ac.logical_z_dimension);
                GPU[tIDX].d_sumSq_ac.CopyDeviceToHost(sumSq_ac, true, false);
                GPU[tIDX].d_sum_cc.CopyDeviceToHost(sum_cc, true, false);
                GPU[tIDX].d_sumSq_cc.CopyDeviceToHost(sumSq_cc, true, false);

                GPU[tIDX].d_max_intensity_projection_ac_all_views.Wait( );
                GPU[tIDX].d_max_intensity_projection_cc_all_views.Wait( );

                // TODO swap max_padding for explicit padding in x/y and limit calcs to that region.
                for ( current_y = 0; current_y < max_intensity_projection_ac.logical_y_dimension; current_y++ ) {
                    for ( current_x = 0; current_x < max_intensity_projection_ac.logical_x_dimension; current_x++ ) {
                        // first mip
                        // TODO add slice number/view counter
                        for ( view_counter = 0; view_counter < total_correlation_positions_sampled_view; view_counter++ ) {
                            if ( mip_buffer_ac.ReturnRealPixelFromPhysicalCoord(current_x, current_y, view_counter) > max_intensity_projection_ac.ReturnRealPixelFromPhysicalCoord(current_x, current_y, view_counter) ) {
                                max_intensity_projection_ac.ReplaceRealPixelAtPhysicalCoord(mip_buffer_ac.ReturnRealPixelFromPhysicalCoord(current_x, current_y, view_counter), current_x, current_y, view_counter);
                            }
                            if ( mip_buffer_cc.ReturnRealPixelFromPhysicalCoord(current_x, current_y, view_counter) > max_intensity_projection_cc.ReturnRealPixelFromPhysicalCoord(current_x, current_y, view_counter) ) {
                                max_intensity_projection_cc.ReplaceRealPixelAtPhysicalCoord(mip_buffer_cc.ReturnRealPixelFromPhysicalCoord(current_x, current_y, view_counter), current_x, current_y, view_counter);
                            }
                            correlation_pixel_sum_ac.ReplaceRealPixelAtPhysicalCoord(correlation_pixel_sum_ac.ReturnRealPixelFromPhysicalCoord(current_x, current_y, view_counter) + sum_ac.ReturnRealPixelFromPhysicalCoord(current_x, current_y, view_counter), current_x, current_y, view_counter);
                            correlation_pixel_sum_of_squares_ac.ReplaceRealPixelAtPhysicalCoord(correlation_pixel_sum_of_squares_ac.ReturnRealPixelFromPhysicalCoord(current_x, current_y, view_counter) + sumSq_ac.ReturnRealPixelFromPhysicalCoord(current_x, current_y, view_counter), current_x, current_y, view_counter);
                            correlation_pixel_sum_cc.ReplaceRealPixelAtPhysicalCoord(correlation_pixel_sum_cc.ReturnRealPixelFromPhysicalCoord(current_x, current_y, view_counter) + sum_cc.ReturnRealPixelFromPhysicalCoord(current_x, current_y, view_counter), current_x, current_y, view_counter);
                            correlation_pixel_sum_of_squares_cc.ReplaceRealPixelAtPhysicalCoord(correlation_pixel_sum_of_squares_cc.ReturnRealPixelFromPhysicalCoord(current_x, current_y, view_counter) + sumSq_cc.ReturnRealPixelFromPhysicalCoord(current_x, current_y, view_counter), current_x, current_y, view_counter);
                        }
                    }
                }

                // GPU[tIDX].histogram.CopyToHostAndAdd(histogram_data);

                //                    current_correlation_position += GPU[tIDX].total_number_of_cccs_calculated;
                actual_number_of_ccs_calculated += GPU[tIDX].total_number_of_cccs_calculated;

            } // end of omp critical block
        } // end of parallel block

#endif
    }
    /*
#pragma omp parallel for num_threads(max_threads) default(shared) private(current_search_position_sampled_view, j, view_counter, angles_sampled_view, current_psi_sampled_view, current_search_position_tm, current_psi_tm, angles_tm, variance, pixel_counter, current_x, current_y, current_projection_other, current_projection_correct_template, current_projection_image, ac_val, ac_max, cc_val, cc_max, padded_projection)
    for ( current_search_position_sampled_view = first_search_position; current_search_position_sampled_view <= last_search_position_sampled_view; current_search_position_sampled_view++ ) {
        for ( j = 0; j < number_of_rotations_sampled_view; j++ ) {
            current_psi_sampled_view = psi_start + j * psi_step_sampled_view;
            view_counter             = current_search_position_sampled_view * number_of_rotations_sampled_view + j;
            if ( padding != 1.0f )
                padded_projection.Allocate(input_reconstruction_correct.logical_x_dimension, input_reconstruction_correct.logical_y_dimension, false);
            current_projection_image.Allocate(input_reconstruction_particle.logical_x_dimension, input_reconstruction_particle.logical_y_dimension, 1, false);
            current_projection_other.Allocate(input_reconstruction_particle.logical_x_dimension, input_reconstruction_particle.logical_y_dimension, 1, false); // use particle to determine size since template volumes may be padded
            current_projection_correct_template.Allocate(input_reconstruction_particle.logical_x_dimension, input_reconstruction_particle.logical_y_dimension, 1, false);
            //wxPrintf("worker #%i working on view #%i\n", ReturnThreadNumberOfCurrentThread( ), view_counter);
            angles_sampled_view.Init(global_euler_search_sampled_view.list_of_search_parameters[current_search_position_sampled_view][0], global_euler_search_sampled_view.list_of_search_parameters[current_search_position_sampled_view][1], current_psi_sampled_view, 0.0, 0.0);
            // generate particle from sampled view
            input_reconstruction_particle.ExtractSlice(current_projection_image, angles_sampled_view, 1.0f, false);
            current_projection_image.SwapRealSpaceQuadrants( );
            current_projection_image.MultiplyPixelWise(projection_filter);
            current_projection_image.BackwardFFT( );
            //wxPrintf("view %i before avg = %f std = %f\n", view_counter, current_projection_image.ReturnAverageOfRealValues( ), sqrtf(current_projection_image.ReturnVarianceOfRealValues( )));
            log_file.WriteCommentLine("view %i before avg = %f std = %f\n", view_counter, current_projection_image.ReturnAverageOfRealValues( ), sqrtf(current_projection_image.ReturnVarianceOfRealValues( )));

            current_projection_image.AddConstant(-img_avg);
            //current_projection_image.ZeroCentralPixel( );
            //current_projection_image.DivideByConstant(sqrtf(current_projection_image.ReturnSumOfSquares( )));
            //current_projection_image.QuickAndDirtyWriteSlice(wxString::Format("view_%i_image.mrc", view_counter).ToStdString( ), 1);
            current_projection_image.DivideByConstant(img_std);
            //wxPrintf("view %i after avg = %f std = %f\n", view_counter, current_projection_image.ReturnAverageOfRealValues( ), current_projection_image.ReturnVarianceOfRealValues( ));
            log_file.WriteCommentLine("view %i after avg = %f std = %f\n", view_counter, current_projection_image.ReturnAverageOfRealValues( ), current_projection_image.ReturnVarianceOfRealValues( ));
            //if ( view_counter == 0 )
            //    current_projection_image.QuickAndDirtyWriteSlice(wxString::Format("view_%i_image_after_scaling.mrc", view_counter).ToStdString( ), 1);

            current_projection_image.ForwardFFT( );

            for ( current_search_position_tm = first_search_position; current_search_position_tm <= last_search_position_tm; current_search_position_tm++ ) {
                for ( current_psi_tm = psi_start; current_psi_tm <= psi_max; current_psi_tm += psi_step_tm ) {
                    angles_tm.Init(global_euler_search_tm.list_of_search_parameters[current_search_position_tm][0], global_euler_search_tm.list_of_search_parameters[current_search_position_tm][1], current_psi_tm, 0.0, 0.0);
                    // generate projection from testing template for tm
                    //input_reconstruction_wrong.ExtractSlice(current_projection_other, angles_tm, 1.0f, false);
                    //current_projection_other.SwapRealSpaceQuadrants( );
                    if ( padding != 1.0f ) {
                        input_reconstruction_wrong.ExtractSlice(padded_projection, angles_tm, 1.0f, false);
                        padded_projection.SwapRealSpaceQuadrants( );
                        padded_projection.BackwardFFT( );
                        padded_projection.ClipInto(&current_projection_other);
                        current_projection_other.ForwardFFT( );
                        padded_projection.ForwardFFT( );
                    }
                    else {
                        input_reconstruction_wrong.ExtractSlice(current_projection_other, angles_tm, 1.0f, false);
                        current_projection_other.SwapRealSpaceQuadrants( );
                    }
                    current_projection_other.MultiplyPixelWise(projection_filter);
                    current_projection_other.BackwardFFT( );
                    variance = current_projection_other.ReturnSumOfSquares( ) - powf(current_projection_other.ReturnAverageOfRealValues( ), 2);
                    current_projection_other.DivideByConstant(sqrtf(variance));
                    //current_projection_2.AddGaussianNoise(10.0f);
                    current_projection_other.ForwardFFT( );
                    // Zeroing the central pixel is probably not doing anything useful...
                    current_projection_other.ZeroCentralPixel( );

                    // generate projection from particle's template for tm
                    //input_reconstruction_correct.ExtractSlice(current_projection_correct_template, angles_tm, 1.0f, false);
                    //current_projection_correct_template.SwapRealSpaceQuadrants( );
                    if ( padding != 1.0f ) {
                        input_reconstruction_correct.ExtractSlice(padded_projection, angles_tm, 1.0f, false);
                        padded_projection.SwapRealSpaceQuadrants( );
                        padded_projection.BackwardFFT( );
                        padded_projection.ClipInto(&current_projection_correct_template);
                        current_projection_correct_template.ForwardFFT( );
                        padded_projection.ForwardFFT( );
                    }
                    else {
                        input_reconstruction_correct.ExtractSlice(current_projection_correct_template, angles_tm, 1.0f, false);
                        current_projection_correct_template.SwapRealSpaceQuadrants( );
                    }

                    current_projection_correct_template.MultiplyPixelWise(projection_filter);
                    current_projection_correct_template.BackwardFFT( );
                    variance = current_projection_correct_template.ReturnSumOfSquares( ) - powf(current_projection_correct_template.ReturnAverageOfRealValues( ), 2);
                    current_projection_correct_template.DivideByConstant(sqrtf(variance));
                    //current_projection_2.AddGaussianNoise(10.0f);
                    current_projection_correct_template.ForwardFFT( );
                    // Zeroing the central pixel is probably not doing anything useful...
                    current_projection_correct_template.ZeroCentralPixel( );

                    //if ( view_counter == 0 ) {

                    //    current_projection_correct_template.QuickAndDirtyWriteSlice(wxString::Format("view_%i_ref1.mrc", view_counter).ToStdString( ), 1);
                    //    current_projection_other.QuickAndDirtyWriteSlice(wxString::Format("view_%i_ref2.mrc", view_counter).ToStdString( ), 1);
                    //}

#ifdef MKL
                    // Use the MKL
                    vmcMulByConj(current_projection_correct_template.real_memory_allocated / 2, reinterpret_cast<MKL_Complex8*>(current_projection_image.complex_values), reinterpret_cast<MKL_Complex8*>(current_projection_correct_template.complex_values), reinterpret_cast<MKL_Complex8*>(current_projection_correct_template.complex_values), VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
#else
                    for ( pixel_counter = 0; pixel_counter < current_projection_image.real_memory_allocated / 2; pixel_counter++ ) {
                        current_projection_correct_template.complex_values[pixel_counter] = conj(current_projection_correct_template.complex_values[pixel_counter]) * current_projection_image.complex_values[pixel_counter];
                    }
#endif

                    current_projection_correct_template.SwapRealSpaceQuadrants( );
                    current_projection_correct_template.BackwardFFT( );
                    ac_val = current_projection_correct_template.ReturnCentralPixelValue( );
                    ac_max = current_projection_correct_template.ReturnMaximumValue( );

#ifdef MKL
                    // Use the MKL
                    vmcMulByConj(current_projection_other.real_memory_allocated / 2, reinterpret_cast<MKL_Complex8*>(current_projection_image.complex_values), reinterpret_cast<MKL_Complex8*>(current_projection_other.complex_values), reinterpret_cast<MKL_Complex8*>(current_projection_other.complex_values), VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
#else
                    for ( pixel_counter = 0; pixel_counter < current_projection_1.real_memory_allocated / 2; pixel_counter++ ) {
                        current_projection_other.complex_values[pixel_counter] = conj(current_projection_other.complex_values[pixel_counter]) * current_projection_image.complex_values[pixel_counter];
                    }
#endif
                    current_projection_other.SwapRealSpaceQuadrants( );
                    current_projection_other.BackwardFFT( );
                    cc_val = current_projection_other.ReturnCentralPixelValue( );
                    cc_max = current_projection_other.ReturnMaximumValue( );

                    // find alignment using ac
                    if ( current_psi_sampled_view == current_psi_tm && global_euler_search_sampled_view.list_of_search_parameters[current_search_position_sampled_view][0] == global_euler_search_tm.list_of_search_parameters[current_search_position_tm][0] && global_euler_search_sampled_view.list_of_search_parameters[current_search_position_sampled_view][1] == global_euler_search_tm.list_of_search_parameters[current_search_position_tm][1] ) {

                        log_file.WriteCommentLine("central ac / max ac = %f %f \n", ac_val, ac_max);
                        log_file.WriteCommentLine("central cc / max cc = %f %f \n", cc_val, cc_max);

                        //    continue;
                    }

                    // loop through the whole image to update each pixel on mip
                    pixel_counter = 0;
                    for ( current_y = 0; current_y < max_intensity_projection_cc.logical_y_dimension; current_y++ ) {
                        for ( current_x = 0; current_x < max_intensity_projection_cc.logical_x_dimension; current_x++ ) {
                            // update mip cc

                            if ( current_projection_other.real_values[pixel_counter] > max_intensity_projection_cc.ReturnRealPixelFromPhysicalCoord(current_x, current_y, view_counter) ) {
                                max_intensity_projection_cc.ReplaceRealPixelAtPhysicalCoord(current_projection_other.real_values[pixel_counter], current_x, current_y, view_counter);
                            }

                            // update mip ac
                            if ( current_projection_correct_template.real_values[pixel_counter] > max_intensity_projection_ac.ReturnRealPixelFromPhysicalCoord(current_x, current_y, view_counter) ) {
                                max_intensity_projection_ac.ReplaceRealPixelAtPhysicalCoord(current_projection_correct_template.real_values[pixel_counter], current_x, current_y, view_counter);
                            }
                            correlation_pixel_sum_ac.real_values[correlation_pixel_sum_ac.ReturnReal1DAddressFromPhysicalCoord(current_x, current_y, view_counter)] += current_projection_correct_template.real_values[pixel_counter];
                            correlation_pixel_sum_cc.real_values[correlation_pixel_sum_cc.ReturnReal1DAddressFromPhysicalCoord(current_x, current_y, view_counter)] += current_projection_other.real_values[pixel_counter];
                            correlation_pixel_sum_of_squares_ac.real_values[correlation_pixel_sum_of_squares_ac.ReturnReal1DAddressFromPhysicalCoord(current_x, current_y, view_counter)] += pow(current_projection_correct_template.real_values[pixel_counter], 2);
                            correlation_pixel_sum_of_squares_cc.real_values[correlation_pixel_sum_of_squares_cc.ReturnReal1DAddressFromPhysicalCoord(current_x, current_y, view_counter)] += pow(current_projection_other.real_values[pixel_counter], 2);

                            pixel_counter++;
                        }
                        pixel_counter += current_projection_other.padding_jump_value;
                    }
                }
            }
        }
    }

    double sqrt_input_pixels = sqrt((double)(input_reconstruction_particle.logical_x_dimension * input_reconstruction_particle.logical_y_dimension));
    wxPrintf("N = %f\n", float(sqrt_input_pixels));
    max_intensity_projection_ac.MultiplyByConstant((float)sqrt_input_pixels);
    max_intensity_projection_cc.MultiplyByConstant((float)sqrt_input_pixels);

    max_intensity_projection_cc.QuickAndDirtyWriteSlices("mip_cc.mrc", 1, total_correlation_positions_sampled_view);
    max_intensity_projection_ac.QuickAndDirtyWriteSlices("mip_ac.mrc", 1, total_correlation_positions_sampled_view);

    correlation_pixel_sum_of_squares_ac.QuickAndDirtyWriteSlices("sos_ac.mrc", 1, total_correlation_positions_sampled_view);
    correlation_pixel_sum_of_squares_cc.QuickAndDirtyWriteSlices("sos_cc.mrc", 1, total_correlation_positions_sampled_view);

    pixel_counter = 0;
    for ( int current_y = 0; current_y < correlation_pixel_sum_ac.logical_y_dimension; current_y++ ) {
        for ( int current_x = 0; current_x < correlation_pixel_sum_ac.logical_x_dimension; current_x++ ) {
            for ( view_counter = 0; view_counter < total_correlation_positions_sampled_view; view_counter++ ) {
                correlation_pixel_sum_ac.real_values[correlation_pixel_sum_ac.ReturnReal1DAddressFromPhysicalCoord(current_x, current_y, view_counter)] /= float(total_correlation_positions_tm);

                correlation_pixel_sum_cc.real_values[correlation_pixel_sum_cc.ReturnReal1DAddressFromPhysicalCoord(current_x, current_y, view_counter)] /= float(total_correlation_positions_tm);

                correlation_pixel_sum_of_squares_ac.real_values[correlation_pixel_sum_of_squares_ac.ReturnReal1DAddressFromPhysicalCoord(current_x, current_y, view_counter)] = correlation_pixel_sum_of_squares_ac.ReturnRealPixelFromPhysicalCoord(current_x, current_y, view_counter) / float(total_correlation_positions_tm) - powf(correlation_pixel_sum_ac.ReturnRealPixelFromPhysicalCoord(current_x, current_y, view_counter), 2);

                correlation_pixel_sum_of_squares_cc.real_values[correlation_pixel_sum_of_squares_cc.ReturnReal1DAddressFromPhysicalCoord(current_x, current_y, view_counter)] = correlation_pixel_sum_of_squares_cc.ReturnRealPixelFromPhysicalCoord(current_x, current_y, view_counter) / float(total_correlation_positions_tm) - powf(correlation_pixel_sum_cc.ReturnRealPixelFromPhysicalCoord(current_x, current_y, view_counter), 2);

                if ( correlation_pixel_sum_of_squares_ac.ReturnRealPixelFromPhysicalCoord(current_x, current_y, view_counter) > 0.0f ) {
                    correlation_pixel_sum_of_squares_ac.real_values[correlation_pixel_sum_of_squares_ac.ReturnReal1DAddressFromPhysicalCoord(current_x, current_y, view_counter)] = sqrtf(correlation_pixel_sum_of_squares_ac.ReturnRealPixelFromPhysicalCoord(current_x, current_y, view_counter)) * (float)sqrt_input_pixels;
                }
                else
                    correlation_pixel_sum_of_squares_ac.ReplaceRealPixelAtPhysicalCoord(0.0f, current_x, current_y, view_counter);

                correlation_pixel_sum_ac.real_values[correlation_pixel_sum_ac.ReturnReal1DAddressFromPhysicalCoord(current_x, current_y, view_counter)] *= (float)sqrt_input_pixels;

                if ( correlation_pixel_sum_of_squares_cc.ReturnRealPixelFromPhysicalCoord(current_x, current_y, view_counter) > 0.0f ) {
                    correlation_pixel_sum_of_squares_cc.real_values[correlation_pixel_sum_of_squares_cc.ReturnReal1DAddressFromPhysicalCoord(current_x, current_y, view_counter)] = sqrtf(correlation_pixel_sum_of_squares_cc.ReturnRealPixelFromPhysicalCoord(current_x, current_y, view_counter)) * (float)sqrt_input_pixels;
                }
                else
                    correlation_pixel_sum_of_squares_cc.ReplaceRealPixelAtPhysicalCoord(0.0f, current_x, current_y, view_counter);

                correlation_pixel_sum_cc.real_values[correlation_pixel_sum_cc.ReturnReal1DAddressFromPhysicalCoord(current_x, current_y, view_counter)] *= (float)sqrt_input_pixels;
            }
            pixel_counter++;
        }
        pixel_counter += current_projection_other.padding_jump_value;
    }
    correlation_pixel_sum_ac.QuickAndDirtyWriteSlices("avg_ac.mrc", 1, total_correlation_positions_sampled_view);
    correlation_pixel_sum_cc.QuickAndDirtyWriteSlices("avg_cc.mrc", 1, total_correlation_positions_sampled_view);

    correlation_pixel_sum_of_squares_ac.QuickAndDirtyWriteSlices("std_ac.mrc", 1, total_correlation_positions_sampled_view);
    correlation_pixel_sum_of_squares_cc.QuickAndDirtyWriteSlices("std_cc.mrc", 1, total_correlation_positions_sampled_view);
    // scaling

    for ( pixel_counter = 0; pixel_counter < max_intensity_projection_ac.real_memory_allocated; pixel_counter++ ) {
        max_intensity_projection_ac.real_values[pixel_counter] -= correlation_pixel_sum_ac.real_values[pixel_counter];
        max_intensity_projection_cc.real_values[pixel_counter] -= correlation_pixel_sum_cc.real_values[pixel_counter];

        if ( correlation_pixel_sum_of_squares_ac.real_values[pixel_counter] > 0.0f ) {
            max_intensity_projection_ac.real_values[pixel_counter] /= correlation_pixel_sum_of_squares_ac.real_values[pixel_counter];
        }
        else
            max_intensity_projection_ac.real_values[pixel_counter] = 0.0f;

        if ( correlation_pixel_sum_of_squares_cc.real_values[pixel_counter] > 0.0f ) {
            max_intensity_projection_cc.real_values[pixel_counter] /= correlation_pixel_sum_of_squares_cc.real_values[pixel_counter];
        }
        else
            max_intensity_projection_cc.real_values[pixel_counter] = 0.0f;
    }

    max_intensity_projection_ac.QuickAndDirtyWriteSlices("scaled_mip_ac.mrc", 1, total_correlation_positions_sampled_view);
    max_intensity_projection_cc.QuickAndDirtyWriteSlices("scaled_mip_cc.mrc", 1, total_correlation_positions_sampled_view);
    */

    // post processing and saving outputs
    double sqrt_input_pixels = sqrt((double)(input_reconstruction_particle.logical_x_dimension * input_reconstruction_particle.logical_y_dimension));
    wxPrintf("N = %f\n", float(sqrt_input_pixels));
    max_intensity_projection_ac.MultiplyByConstant((float)sqrt_input_pixels);
    max_intensity_projection_cc.MultiplyByConstant((float)sqrt_input_pixels);
    max_intensity_projection_ac.QuickAndDirtyWriteSlices(mip_ac_filename.ToStdString( ), 1, total_correlation_positions_sampled_view);
    max_intensity_projection_cc.QuickAndDirtyWriteSlices(mip_cc_filename.ToStdString( ), 1, total_correlation_positions_sampled_view);

    //  correlation_pixel_sum_of_squares_ac.QuickAndDirtyWriteSlices("check_gpu_run/sos_ac.mrc", 1, total_correlation_positions_sampled_view);
    //  correlation_pixel_sum_of_squares_cc.QuickAndDirtyWriteSlices("check_gpu_run/sos_cc.mrc", 1, total_correlation_positions_sampled_view);

    //  correlation_pixel_sum_ac.QuickAndDirtyWriteSlices("check_gpu_run/sum_ac.mrc", 1, total_correlation_positions_sampled_view);
    //  correlation_pixel_sum_cc.QuickAndDirtyWriteSlices("check_gpu_run/sum_cc.mrc", 1, total_correlation_positions_sampled_view);

    for ( int current_y = 0; current_y < correlation_pixel_sum_ac.logical_y_dimension; current_y++ ) {
        for ( int current_x = 0; current_x < correlation_pixel_sum_ac.logical_x_dimension; current_x++ ) {
            for ( view_counter = 0; view_counter < total_correlation_positions_sampled_view; view_counter++ ) {
                correlation_pixel_sum_ac.real_values[correlation_pixel_sum_ac.ReturnReal1DAddressFromPhysicalCoord(current_x, current_y, view_counter)] /= float(total_correlation_positions_tm);

                correlation_pixel_sum_cc.real_values[correlation_pixel_sum_cc.ReturnReal1DAddressFromPhysicalCoord(current_x, current_y, view_counter)] /= float(total_correlation_positions_tm);

                correlation_pixel_sum_of_squares_ac.real_values[correlation_pixel_sum_of_squares_ac.ReturnReal1DAddressFromPhysicalCoord(current_x, current_y, view_counter)] = correlation_pixel_sum_of_squares_ac.ReturnRealPixelFromPhysicalCoord(current_x, current_y, view_counter) / float(total_correlation_positions_tm) - powf(correlation_pixel_sum_ac.ReturnRealPixelFromPhysicalCoord(current_x, current_y, view_counter), 2);

                correlation_pixel_sum_of_squares_cc.real_values[correlation_pixel_sum_of_squares_cc.ReturnReal1DAddressFromPhysicalCoord(current_x, current_y, view_counter)] = correlation_pixel_sum_of_squares_cc.ReturnRealPixelFromPhysicalCoord(current_x, current_y, view_counter) / float(total_correlation_positions_tm) - powf(correlation_pixel_sum_cc.ReturnRealPixelFromPhysicalCoord(current_x, current_y, view_counter), 2);

                if ( correlation_pixel_sum_of_squares_ac.ReturnRealPixelFromPhysicalCoord(current_x, current_y, view_counter) > 0.0f ) {
                    correlation_pixel_sum_of_squares_ac.real_values[correlation_pixel_sum_of_squares_ac.ReturnReal1DAddressFromPhysicalCoord(current_x, current_y, view_counter)] = sqrtf(correlation_pixel_sum_of_squares_ac.ReturnRealPixelFromPhysicalCoord(current_x, current_y, view_counter)) * (float)sqrt_input_pixels;
                }
                else
                    correlation_pixel_sum_of_squares_ac.ReplaceRealPixelAtPhysicalCoord(0.0f, current_x, current_y, view_counter);

                correlation_pixel_sum_ac.real_values[correlation_pixel_sum_ac.ReturnReal1DAddressFromPhysicalCoord(current_x, current_y, view_counter)] *= (float)sqrt_input_pixels;

                if ( correlation_pixel_sum_of_squares_cc.ReturnRealPixelFromPhysicalCoord(current_x, current_y, view_counter) > 0.0f ) {
                    correlation_pixel_sum_of_squares_cc.real_values[correlation_pixel_sum_of_squares_cc.ReturnReal1DAddressFromPhysicalCoord(current_x, current_y, view_counter)] = sqrtf(correlation_pixel_sum_of_squares_cc.ReturnRealPixelFromPhysicalCoord(current_x, current_y, view_counter)) * (float)sqrt_input_pixels;
                }
                else
                    correlation_pixel_sum_of_squares_cc.ReplaceRealPixelAtPhysicalCoord(0.0f, current_x, current_y, view_counter);

                correlation_pixel_sum_cc.real_values[correlation_pixel_sum_cc.ReturnReal1DAddressFromPhysicalCoord(current_x, current_y, view_counter)] *= (float)sqrt_input_pixels;
            }
        }
    }
    correlation_pixel_sum_ac.QuickAndDirtyWriteSlices(avg_ac_filename.ToStdString( ), 1, total_correlation_positions_sampled_view);
    correlation_pixel_sum_cc.QuickAndDirtyWriteSlices(avg_cc_filename.ToStdString( ), 1, total_correlation_positions_sampled_view);

    correlation_pixel_sum_of_squares_ac.QuickAndDirtyWriteSlices(std_ac_filename.ToStdString( ), 1, total_correlation_positions_sampled_view);
    correlation_pixel_sum_of_squares_cc.QuickAndDirtyWriteSlices(std_cc_filename.ToStdString( ), 1, total_correlation_positions_sampled_view);
    // scaling

    for ( pixel_counter = 0; pixel_counter < max_intensity_projection_ac.real_memory_allocated; pixel_counter++ ) {
        max_intensity_projection_ac.real_values[pixel_counter] -= correlation_pixel_sum_ac.real_values[pixel_counter];
        max_intensity_projection_cc.real_values[pixel_counter] -= correlation_pixel_sum_cc.real_values[pixel_counter];

        if ( correlation_pixel_sum_of_squares_ac.real_values[pixel_counter] > 0.0f ) {
            max_intensity_projection_ac.real_values[pixel_counter] /= correlation_pixel_sum_of_squares_ac.real_values[pixel_counter];
        }
        else
            max_intensity_projection_ac.real_values[pixel_counter] = 0.0f;

        if ( correlation_pixel_sum_of_squares_cc.real_values[pixel_counter] > 0.0f ) {
            max_intensity_projection_cc.real_values[pixel_counter] /= correlation_pixel_sum_of_squares_cc.real_values[pixel_counter];
        }
        else
            max_intensity_projection_cc.real_values[pixel_counter] = 0.0f;
    }

    max_intensity_projection_ac.QuickAndDirtyWriteSlices(scaled_mip_ac_filename.ToStdString( ), 1, total_correlation_positions_sampled_view);
    max_intensity_projection_cc.QuickAndDirtyWriteSlices(scaled_mip_cc_filename.ToStdString( ), 1, total_correlation_positions_sampled_view);
#ifdef ENABLEGPU
    if ( use_gpu ) {
        delete[] GPU;
    }

#endif

    wxPrintf("\n\n\tTimings: Overall: %s\n", (wxDateTime::Now( ) - overall_start).Format( ));

    return true;
}

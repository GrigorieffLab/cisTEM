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
    wxString output_pose_filename;
    wxString best_psi_ac_filename;
    wxString best_theta_ac_filename;
    wxString best_phi_ac_filename;
    wxString best_psi_cc_filename;
    wxString best_theta_cc_filename;
    wxString best_phi_cc_filename;

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
    int      number_of_sampled_views        = 30;
    int      result_number;

    UserInput* my_input = new UserInput("CompareTemplate", 1.00);

    input_search_images                    = my_input->GetFilenameFromUser("Input images to be searched", "The input image stack, containing the images that should be searched", "image_stack.mrc", true);
    input_reconstruction_particle_filename = my_input->GetFilenameFromUser("Input template reconstruction for simulating particles", "The 3D reconstruction from which projections are calculated", "reconstruction.mrc", true);
    input_reconstruction_correct_filename  = my_input->GetFilenameFromUser("Correct input template reconstruction", "The 3D reconstruction from which projections are calculated", "reconstruction.mrc", true);
    input_reconstruction_wrong_filename    = my_input->GetFilenameFromUser("Wrong input template reconstruction", "The 3D reconstruction from which projections are calculated", "reconstruction.mrc", true);
    output_pose_filename                   = my_input->GetFilenameFromUser("Output filename for poses of sampled views", "Output filename for poses of sampled views", "pose.star", false);
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
    scaled_mip_ac_filename  = my_input->GetFilenameFromUser("Output scaled mip image ac", "Output scaled mip image ac", "scaled_mip_ac.mrc", false);
    scaled_mip_cc_filename  = my_input->GetFilenameFromUser("Output scaled mip image cc", "Output scaled mip image ac", "scaled_mip_cc.mrc", false);
    mip_ac_filename         = my_input->GetFilenameFromUser("Output mip image ac", "Output mip image ac", "mip_ac.mrc", false);
    mip_cc_filename         = my_input->GetFilenameFromUser("Output mip image cc", "Output mip image cc", "mip_cc.mrc", false);
    avg_ac_filename         = my_input->GetFilenameFromUser("Output avg image ac", "Output avg image ac", "avg_ac.mrc", false);
    avg_cc_filename         = my_input->GetFilenameFromUser("Output avg image cc", "Output avg image cc", "avg_cc.mrc", false);
    std_ac_filename         = my_input->GetFilenameFromUser("Output std image ac", "Output std image ac", "std_cc.mrc", false);
    std_cc_filename         = my_input->GetFilenameFromUser("Output std image cc", "Output std image cc", "std_cc.mrc", false);
    best_psi_ac_filename    = my_input->GetFilenameFromUser("Output autocorrelation psi file", "The file for saving the best psi image", "psi.mrc", false);
    best_theta_ac_filename  = my_input->GetFilenameFromUser("Output autocorrelation theta file", "The file for saving the best theta image", "theta.mrc", false);
    best_phi_ac_filename    = my_input->GetFilenameFromUser("Output autocorrelation phi file", "The file for saving the best phi image", "phi.mrc", false);
    best_psi_cc_filename    = my_input->GetFilenameFromUser("Output cross-correlation psi file", "The file for saving the best psi image", "psi.mrc", false);
    best_theta_cc_filename  = my_input->GetFilenameFromUser("Output cross-correlation theta file", "The file for saving the best theta image", "theta.mrc", false);
    best_phi_cc_filename    = my_input->GetFilenameFromUser("Output cross-correlation phi file", "The file for saving the best phi image", "phi.mrc", false);
    data_directory_name     = my_input->GetFilenameFromUser("Name for data directory", "path to data directory", "60_120_5_2.5", false);
    number_of_sampled_views = my_input->GetIntFromUser("Number of sampled views", "number of sampled views", "1", 1, 400);
    result_number           = my_input->GetIntFromUser("Result number", "result number", "1", 1, 400);

    int first_search_position             = -1;
    int last_search_position_sampled_view = -1;
    int last_search_position_tm           = -1;

    delete my_input;

    my_current_job.ManualSetArguments("tttttfffffffffifftfiiffitbitttttttttttttttii",
                                      input_search_images.ToUTF8( ).data( ),
                                      input_reconstruction_particle_filename.ToUTF8( ).data( ),
                                      input_reconstruction_correct_filename.ToUTF8( ).data( ),
                                      input_reconstruction_wrong_filename.ToUTF8( ).data( ),
                                      output_pose_filename.ToUTF8( ).data( ),
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
                                      best_psi_ac_filename.ToUTF8( ).data( ),
                                      best_theta_ac_filename.ToUTF8( ).data( ),
                                      best_phi_ac_filename.ToUTF8( ).data( ),
                                      best_psi_cc_filename.ToUTF8( ).data( ),
                                      best_theta_cc_filename.ToUTF8( ).data( ),
                                      best_phi_cc_filename.ToUTF8( ).data( ),
                                      data_directory_name.ToUTF8( ).data( ),
                                      number_of_sampled_views,
                                      result_number);
}

// override the do calculation method which will be what is actually run..

bool CompareTemplateApp::DoCalculation( ) {
    wxString input_search_images_filename           = my_current_job.arguments[0].ReturnStringArgument( );
    wxString input_reconstruction_particle_filename = my_current_job.arguments[1].ReturnStringArgument( );
    wxString input_reconstruction_correct_filename  = my_current_job.arguments[2].ReturnStringArgument( );
    wxString input_reconstruction_wrong_filename    = my_current_job.arguments[3].ReturnStringArgument( );
    wxString output_pose_filename                   = my_current_job.arguments[4].ReturnStringArgument( );
    float    pixel_size                             = my_current_job.arguments[5].ReturnFloatArgument( );
    float    voltage_kV                             = my_current_job.arguments[6].ReturnFloatArgument( );
    float    spherical_aberration_mm                = my_current_job.arguments[7].ReturnFloatArgument( );
    float    amplitude_contrast                     = my_current_job.arguments[8].ReturnFloatArgument( );
    float    defocus1                               = my_current_job.arguments[9].ReturnFloatArgument( );
    float    defocus2                               = my_current_job.arguments[10].ReturnFloatArgument( );
    float    defocus_angle                          = my_current_job.arguments[11].ReturnFloatArgument( );
    ;
    float    high_resolution_limit_search      = my_current_job.arguments[12].ReturnFloatArgument( );
    float    angular_step_sampling             = my_current_job.arguments[13].ReturnFloatArgument( );
    int      best_parameters_to_keep           = my_current_job.arguments[14].ReturnIntegerArgument( );
    float    padding                           = my_current_job.arguments[15].ReturnFloatArgument( );
    float    phase_shift                       = my_current_job.arguments[16].ReturnFloatArgument( );
    wxString my_symmetry                       = my_current_job.arguments[17].ReturnStringArgument( );
    float    in_plane_angular_step_sampling    = my_current_job.arguments[18].ReturnFloatArgument( );
    int      first_search_position             = my_current_job.arguments[19].ReturnIntegerArgument( );
    int      last_search_position_sampled_view = my_current_job.arguments[20].ReturnIntegerArgument( );
    float    angular_step_tm                   = my_current_job.arguments[21].ReturnFloatArgument( );
    float    in_plane_angular_step_tm          = my_current_job.arguments[22].ReturnFloatArgument( );
    int      last_search_position_tm           = my_current_job.arguments[23].ReturnIntegerArgument( );
    wxString log_output_file                   = my_current_job.arguments[24].ReturnStringArgument( );
    bool     use_gpu                           = my_current_job.arguments[25].ReturnBoolArgument( );
    int      max_threads                       = my_current_job.arguments[26].ReturnIntegerArgument( );
    wxString scaled_mip_ac_filename            = my_current_job.arguments[27].ReturnStringArgument( );
    wxString scaled_mip_cc_filename            = my_current_job.arguments[28].ReturnStringArgument( );
    wxString mip_ac_filename                   = my_current_job.arguments[29].ReturnStringArgument( );
    wxString mip_cc_filename                   = my_current_job.arguments[30].ReturnStringArgument( );
    wxString avg_ac_filename                   = my_current_job.arguments[31].ReturnStringArgument( );
    wxString avg_cc_filename                   = my_current_job.arguments[32].ReturnStringArgument( );
    wxString std_ac_filename                   = my_current_job.arguments[33].ReturnStringArgument( );
    wxString std_cc_filename                   = my_current_job.arguments[34].ReturnStringArgument( );
    wxString best_psi_ac_filename              = my_current_job.arguments[35].ReturnStringArgument( );
    wxString best_theta_ac_filename            = my_current_job.arguments[36].ReturnStringArgument( );
    wxString best_phi_ac_filename              = my_current_job.arguments[37].ReturnStringArgument( );
    wxString best_psi_cc_filename              = my_current_job.arguments[38].ReturnStringArgument( );
    wxString best_theta_cc_filename            = my_current_job.arguments[39].ReturnStringArgument( );
    wxString best_phi_cc_filename              = my_current_job.arguments[40].ReturnStringArgument( );
    wxString data_directory_name               = my_current_job.arguments[41].ReturnStringArgument( );
    int      number_of_sampled_views           = my_current_job.arguments[42].ReturnIntegerArgument( );
    int      result_number                     = my_current_job.arguments[43].ReturnIntegerArgument( );

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
    long total_correlation_positions_tm_per_thread;
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

    Curve           whitening_filter;
    Curve           number_of_terms;
    NumericTextFile log_file(log_output_file, OPEN_TO_WRITE, 1);

    Image           input_reconstruction_particle, input_reconstruction_correct, input_reconstruction_wrong;
    ImageFile       input_search_image_file;
    ImageFile       input_reconstruction_particle_file, input_reconstruction_correct_template_file, input_reconstruction_wrong_template_file;
    Image           input_image;
    Image           montage_image, montage_image_stack;
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

    float* psi_tm;
    float* theta_tm;
    float* phi_tm;

    for ( current_search_position_tm = first_search_position; current_search_position_tm <= last_search_position_tm; current_search_position_tm++ ) {
        //loop over each rotation angle

        for ( current_psi_tm = psi_start; current_psi_tm <= psi_max; current_psi_tm += psi_step_tm ) {
            total_correlation_positions_tm++;
        }
    }

    for ( current_psi_tm = psi_start; current_psi_tm <= psi_max; current_psi_tm += psi_step_tm ) {
        number_of_rotations_tm++;
    }

    wxPrintf("TM step sizes:\n");
    wxPrintf("Searching %i positions on the Euler sphere (first-last: %i-%i)\n", last_search_position_tm - first_search_position, first_search_position, last_search_position_tm);
    wxPrintf("Searching %i rotations per position.\n", number_of_rotations_tm);
    wxPrintf("There are %li correlation positions total.\n\n", total_correlation_positions_tm);

    psi_tm   = new float[number_of_rotations_tm];
    theta_tm = new float[last_search_position_tm - first_search_position + 1];
    phi_tm   = new float[last_search_position_tm - first_search_position + 1];

    for ( int k = 0; k < number_of_rotations_tm; k++ ) {
        psi_tm[k] = psi_start + psi_step_tm * k;
        //   wxPrintf("psi = %f\n", psi_tm[k]);
    }
    for ( int k = first_search_position; k <= last_search_position_tm; k++ ) {
        phi_tm[k]   = global_euler_search_tm.list_of_search_parameters[k][0];
        theta_tm[k] = global_euler_search_tm.list_of_search_parameters[k][1];
        //   wxPrintf("theta phi = %f %f\n", phi_tm[k], theta_tm[k]);
    }

    // These vars are only needed in the GPU code, but also need to be set out here to compile.
    // update GPU setup to "inner" loop only - for tm search
    bool first_gpu_loop = true;
    int  nGPUs          = 1;
    int  nJobs          = last_search_position_tm - first_search_position + 1;
    if ( use_gpu && max_threads > nJobs ) {
        wxPrintf("\n\tWarning, you request more threads (%d) than there are search positions (%d)\n", max_threads, nJobs);
        max_threads = nJobs;
    }

    int minPos = first_search_position;
    int maxPos = last_search_position_tm;
    int incPos = (nJobs) / (max_threads);

#ifdef ENABLEGPU
    TemplateSnrRatioCore* GPU;
    DeviceManager         gpuDev;
#endif

    if ( use_gpu ) {
        total_correlation_positions_tm_per_thread = total_correlation_positions_tm / max_threads;

#ifdef ENABLEGPU
        //    checkCudaErrors(cudaGetDeviceCount(&nGPUs));
        GPU = new TemplateSnrRatioCore[max_threads];
        gpuDev.Init(nGPUs);

#endif
    }

    ProgressBar* my_progress;
    my_progress = new ProgressBar(total_correlation_positions_tm_per_thread);

    // calculate whitening filter and CTF from experimental image
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

    CTF   input_ctf;
    Image projection_filter;
    projection_filter.Allocate(input_reconstruction_particle_file.ReturnXSize( ), input_reconstruction_particle_file.ReturnXSize( ), false);
    input_ctf.Init(voltage_kV, spherical_aberration_mm, amplitude_contrast, defocus1, defocus2, defocus_angle, 0.0, 0.0, 0.0, pixel_size, deg_2_rad(phase_shift));
    input_ctf.SetDefocus(defocus1 / pixel_size, defocus2 / pixel_size, deg_2_rad(defocus_angle));
    projection_filter.CalculateCTFImage(input_ctf);
    projection_filter.ApplyCurveFilter(&whitening_filter);

    // pad 3d volumes
    if ( padding != 1.0f ) {
        input_reconstruction_particle.Resize(input_reconstruction_particle.logical_x_dimension * padding, input_reconstruction_particle.logical_y_dimension * padding, input_reconstruction_particle.logical_z_dimension * padding, input_reconstruction_particle.ReturnAverageOfRealValuesOnEdges( ));
    }
    input_reconstruction_particle.ForwardFFT( );
    input_reconstruction_particle.ZeroCentralPixel( );
    input_reconstruction_particle.SwapRealSpaceQuadrants( );

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

    montage_image_stack.Allocate(input_reconstruction_particle_file.ReturnXSize( ), input_reconstruction_particle_file.ReturnXSize( ), number_of_sampled_views);

    int   j = 0, view_counter = 0;
    int   current_x, current_y;
    Image current_projection_image, current_projection_other, current_projection_correct_template, padded_projection_image;
    float ac_val, ac_max, cc_val, cc_max;

    NumericTextFile pose_file(output_pose_filename, OPEN_TO_WRITE, 3);
    pose_file.WriteCommentLine("Psi, Theta, Phi");
    double temp_double_array[3];

    // generate particle montage
    current_projection_image.Allocate(input_reconstruction_particle_file.ReturnXSize( ), input_reconstruction_particle_file.ReturnXSize( ), 1, false);
    if ( padding != 1.0f )
        padded_projection_image.Allocate(input_reconstruction_particle_file.ReturnXSize( ) * padding, input_reconstruction_particle_file.ReturnXSize( ) * padding, false);
    for ( int K = 0; K < number_of_sampled_views; K++ ) {
        int k1 = rand( ) % number_of_rotations_tm;
        int k2 = rand( ) % (last_search_position_tm - first_search_position + 1);
        wxPrintf("k1 = %i k2 = %i\n", k1, k2);
        temp_double_array[0] = psi_tm[k1];
        temp_double_array[1] = theta_tm[k2];
        temp_double_array[2] = phi_tm[k2];
        pose_file.WriteLine(temp_double_array);
        wxPrintf("sampling view @ %f %f %f\n", phi_tm[k2], theta_tm[k2], psi_tm[k1]);

        angles_sampled_view.Init(phi_tm[k2], theta_tm[k2], psi_tm[k1], 0.0, 0.0);
        // generate particle from sampled view (in TM we applied projection filter first then normalized the templates)
        if ( padding != 1.0f ) {
            input_reconstruction_particle.ExtractSlice(padded_projection_image, angles_sampled_view, 1.0f, false);
            padded_projection_image.SwapRealSpaceQuadrants( );
            padded_projection_image.BackwardFFT( );
            padded_projection_image.ClipInto(&current_projection_image);
            current_projection_image.ForwardFFT( );
        }
        else {
            input_reconstruction_particle.ExtractSlice(current_projection_image, angles_sampled_view, 1.0f, false);
            current_projection_image.SwapRealSpaceQuadrants( );
        }
        current_projection_image.MultiplyPixelWise(projection_filter);
        current_projection_image.BackwardFFT( );
        montage_image_stack.InsertOtherImageAtSpecifiedSlice(&current_projection_image, K);
    }

    pose_file.Close( );
    montage_image_stack.QuickAndDirtyWriteSlices(wxString::Format("%s/montage_stack_%f_%f_%i_%i.mrc", data_directory_name, psi_step_tm, angular_step_tm, number_of_sampled_views, result_number).ToStdString( ), 1, number_of_sampled_views);
    // convert stack to montage
    int montage_dim_x = montage_image_stack.logical_x_dimension * int(sqrt(number_of_sampled_views)); //(last_search_position_sampled_view - first_search_position + 1);
    int montage_dim_y = montage_image_stack.logical_y_dimension * int(sqrt(number_of_sampled_views)); //number_of_rotations_sampled_view;
    montage_image.Allocate(montage_dim_x, montage_dim_y, true);
    montage_image.SetToConstant(0.0f);
    double* correlation_pixel_sum_ac            = new double[montage_image.real_memory_allocated];
    double* correlation_pixel_sum_of_squares_ac = new double[montage_image.real_memory_allocated];
    double* correlation_pixel_sum_cc            = new double[montage_image.real_memory_allocated];
    double* correlation_pixel_sum_of_squares_cc = new double[montage_image.real_memory_allocated];

    ZeroDoubleArray(correlation_pixel_sum_ac, montage_image.real_memory_allocated);
    ZeroDoubleArray(correlation_pixel_sum_of_squares_ac, montage_image.real_memory_allocated);
    ZeroDoubleArray(correlation_pixel_sum_cc, montage_image.real_memory_allocated);
    ZeroDoubleArray(correlation_pixel_sum_of_squares_cc, montage_image.real_memory_allocated);

    // images for storing mip and angular outputs
    Image max_intensity_projection_ac, max_intensity_projection_cc;
    max_intensity_projection_ac.Allocate(montage_dim_x, montage_dim_y, true);
    max_intensity_projection_ac.SetToConstant(0.0f);
    max_intensity_projection_cc.Allocate(montage_dim_x, montage_dim_y, true);
    max_intensity_projection_cc.SetToConstant(0.0f);

    Image correlation_pixel_sum_ac_image, correlation_pixel_sum_of_squares_ac_image, correlation_pixel_sum_cc_image, correlation_pixel_sum_of_squares_cc_image;
    correlation_pixel_sum_ac_image.Allocate(montage_dim_x, montage_dim_y, true);
    correlation_pixel_sum_of_squares_ac_image.Allocate(montage_dim_x, montage_dim_y, true);
    correlation_pixel_sum_cc_image.Allocate(montage_dim_x, montage_dim_y, true);
    correlation_pixel_sum_of_squares_cc_image.Allocate(montage_dim_x, montage_dim_y, true);

    Image best_psi_ac, best_psi_cc, best_theta_ac, best_theta_cc, best_phi_ac, best_phi_cc;
    best_psi_ac.Allocate(montage_dim_x, montage_dim_y, true);
    best_psi_ac.SetToConstant(0.0f);
    best_psi_cc.Allocate(montage_dim_x, montage_dim_y, true);
    best_psi_cc.SetToConstant(0.0f);

    best_theta_ac.Allocate(montage_dim_x, montage_dim_y, true);
    best_theta_ac.SetToConstant(0.0f);
    best_theta_cc.Allocate(montage_dim_x, montage_dim_y, true);
    best_theta_cc.SetToConstant(0.0f);

    best_phi_ac.Allocate(montage_dim_x, montage_dim_y, true);
    best_phi_ac.SetToConstant(0.0f);
    best_phi_cc.Allocate(montage_dim_x, montage_dim_y, true);
    best_phi_cc.SetToConstant(0.0f);

    Image input_slice;
    int   image_counter_x;
    int   image_counter_y;
    int   j_input;
    int   i_input;
    long  counter_input;
    int   i_start_output;
    int   j_start_output;
    float attenuation_x;
    float attenuation_y;

    int       image_counter = 0;
    ImageFile montage_stack_file;
    montage_stack_file.OpenFile(wxString::Format("%s/montage_stack_%f_%f_%i_%i.mrc", data_directory_name, psi_step_tm, angular_step_tm, number_of_sampled_views, result_number).ToStdString( ), false);
    for ( image_counter_y = 0; image_counter_y < int(sqrt(number_of_sampled_views)); image_counter_y++ ) {
        j_start_output = image_counter_y * montage_image_stack.logical_y_dimension;

        for ( image_counter_x = 0; image_counter_x < int(sqrt(number_of_sampled_views)); image_counter_x++ ) {

            i_start_output = image_counter_x * montage_image_stack.logical_x_dimension;

            image_counter++;
            input_slice.ReadSlice(&montage_stack_file, image_counter);

            // Loop over the input image
            counter_input = 0;
            int overlap   = 0;
            for ( j_input = 0; j_input < input_slice.logical_y_dimension; j_input++ ) {
                attenuation_y = 1.0;

                if ( j_input < overlap ) {
                    if ( image_counter_y > 0 )
                        attenuation_y = 1.0 / float(overlap + 1) * (j_input + 1);
                }
                else if ( j_input >= input_slice.logical_y_dimension - overlap ) {
                    if ( image_counter_y < int(sqrt(number_of_sampled_views)) - 1 )
                        attenuation_y = 1.0 / float(overlap + 1) * (input_slice.logical_y_dimension - j_input);
                }

                for ( i_input = 0; i_input < input_slice.logical_x_dimension; i_input++ ) {
                    attenuation_x = 1.0;

                    if ( i_input < overlap ) {
                        if ( image_counter_x > 0 )
                            attenuation_x = 1.0 / float(overlap + 1) * (i_input + 1);
                    }
                    else if ( i_input >= input_slice.logical_x_dimension - overlap ) {
                        if ( image_counter_x < int(sqrt(number_of_sampled_views)) - 1 )
                            attenuation_x = 1.0 / float(overlap + 1) * (input_slice.logical_x_dimension - i_input);
                    }

                    montage_image.real_values[montage_image.ReturnReal1DAddressFromPhysicalCoord(i_start_output + i_input, j_start_output + j_input, 0)] += input_slice.real_values[counter_input] * attenuation_x * attenuation_y;
                    counter_input++;
                }
                counter_input += input_slice.padding_jump_value;
            }
        }
    }
    montage_image.QuickAndDirtyWriteSlice(wxString::Format("%s/montage_%f_%f_%i_%i.mrc", data_directory_name, psi_step_tm, angular_step_tm, number_of_sampled_views, result_number).ToStdString( ), 1);

    // normalize stitched image

    montage_image.ReplaceOutliersWithMean(5.0f);
    montage_image.ForwardFFT( );
    montage_image.SwapRealSpaceQuadrants( ); // this will disrupt image structure but it should be fine?

    montage_image.ZeroCentralPixel( ); // equivalent to subtracting mean in real space

    montage_image.DivideByConstant(sqrtf(montage_image.ReturnSumOfSquares( )));
    montage_image.QuickAndDirtyWriteSlice(wxString::Format("%s/montage_scaled_%f_%f_%i_%i.mrc", data_directory_name, psi_step_tm, angular_step_tm, number_of_sampled_views, result_number).ToStdString( ), 1);

    /*
    // test average cc and std cc Dec 15/2022
    ImageFile input_3d_mean_file;
    input_3d_mean_file.OpenFile("LSU_mean.mrc", false);
    Image template_mean;
    template_mean.ReadSlice(&input_3d_mean_file, 1);

    template_mean.ForwardFFT( );

    template_mean.complex_values[0] = 0.0f + I * 0.0f;

    // template_mean.SwapRealSpaceQuadrants( );

    template_mean.MultiplyPixelWise(projection_filter);
    template_mean.BackwardFFT( );
    template_mean.QuickAndDirtyWriteSlice("template_mean.mrc", 1);
    float avg_on_edge  = template_mean.ReturnAverageOfRealValuesOnEdges( );
    float avg_of_reals = template_mean.ReturnAverageOfRealValues( );
    template_mean.AddConstant(-avg_on_edge);

    Image padded_reference;
    padded_reference.Allocate(montage_image.logical_x_dimension, montage_image.logical_y_dimension, montage_image.logical_z_dimension, true);
    padded_reference.SetToConstant(0.0f);
    avg_of_reals *= template_mean.number_of_real_space_pixels / padded_reference.number_of_real_space_pixels;
    float var = template_mean.ReturnSumOfSquares( ) * template_mean.number_of_real_space_pixels / padded_reference.number_of_real_space_pixels - avg_of_reals * avg_of_reals;
    template_mean.DivideByConstant(sqrtf(var));
    wxPrintf("padded ref dim = %i %i %i\n", padded_reference.logical_x_dimension, padded_reference.logical_y_dimension, padded_reference.logical_z_dimension);
    wxPrintf("template ref dim = %i %i %i\n", template_mean.logical_x_dimension, template_mean.logical_y_dimension, template_mean.logical_z_dimension);

    template_mean.ClipIntoLargerRealSpace2D(&padded_reference);
    padded_reference.ForwardFFT( );
    padded_reference.ZeroCentralPixel( );
    padded_reference.QuickAndDirtyWriteSlice("padded_ref.mrc", 1);
#ifdef MKL
    // Use the MKL
    vmcMulByConj(padded_reference.real_memory_allocated / 2, reinterpret_cast<MKL_Complex8*>(montage_image.complex_values), reinterpret_cast<MKL_Complex8*>(padded_reference.complex_values), reinterpret_cast<MKL_Complex8*>(padded_reference.complex_values), VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
#else
    for ( pixel_counter = 0; pixel_counter < padded_reference.real_memory_allocated / 2; pixel_counter++ ) {
        padded_reference.complex_values[pixel_counter] = conj(padded_reference.complex_values[pixel_counter]) * montage_image.complex_values[pixel_counter];
    }
#endif
    padded_reference.BackwardFFT( );
    wxPrintf("max = %f central = %f\n", padded_reference.ReturnMaximumValue( ), padded_reference.ReturnCentralPixelValue( ));
    double sqrt_input_pixels = sqrt((double)(montage_image.logical_x_dimension * montage_image.logical_y_dimension));
    wxPrintf("N = %f\n", float(sqrt_input_pixels));
    padded_reference.MultiplyByConstant(sqrt_input_pixels);
    padded_reference.QuickAndDirtyWriteSlice("first_cc.mrc", 1);
    exit(0);
*/
    // allocate current_projection before gpu code
    // use particle to determine size since template volumes may be padded
    current_projection_other.Allocate(input_reconstruction_particle_file.ReturnXSize( ), input_reconstruction_particle_file.ReturnXSize( ), 1, false);
    current_projection_correct_template.Allocate(input_reconstruction_particle_file.ReturnXSize( ), input_reconstruction_particle_file.ReturnXSize( ), 1, false);

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

                GPU[tIDX].Init(this, input_reconstruction_correct, input_reconstruction_wrong, montage_image, current_projection_correct_template, current_projection_other, psi_max, psi_start, psi_step_tm, angles_tm, global_euler_search_tm, t_first_search_position, t_last_search_position, my_progress, number_of_rotations_tm, total_correlation_positions_tm, total_correlation_positions_tm_per_thread, data_directory_name);

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
                mip_buffer_cc.CopyFrom(&max_intensity_projection_ac);
                Image psi_buffer_ac, psi_buffer_cc;
                psi_buffer_ac.CopyFrom(&max_intensity_projection_ac);
                psi_buffer_cc.CopyFrom(&max_intensity_projection_ac);
                Image phi_buffer_ac, phi_buffer_cc;
                phi_buffer_ac.CopyFrom(&max_intensity_projection_ac);
                phi_buffer_cc.CopyFrom(&max_intensity_projection_ac);
                Image theta_buffer_ac, theta_buffer_cc;
                theta_buffer_ac.CopyFrom(&max_intensity_projection_ac);
                theta_buffer_cc.CopyFrom(&max_intensity_projection_ac);

                GPU[tIDX].d_max_intensity_projection_ac.CopyDeviceToHost(mip_buffer_ac, true, false);
                GPU[tIDX].d_max_intensity_projection_cc.CopyDeviceToHost(mip_buffer_cc, true, false);
                GPU[tIDX].d_best_psi_ac.CopyDeviceToHost(psi_buffer_ac, true, false);
                GPU[tIDX].d_best_psi_cc.CopyDeviceToHost(psi_buffer_cc, true, false);
                GPU[tIDX].d_best_phi_ac.CopyDeviceToHost(phi_buffer_ac, true, false);
                GPU[tIDX].d_best_phi_cc.CopyDeviceToHost(phi_buffer_cc, true, false);
                GPU[tIDX].d_best_theta_ac.CopyDeviceToHost(theta_buffer_ac, true, false);
                GPU[tIDX].d_best_theta_cc.CopyDeviceToHost(theta_buffer_cc, true, false);

                // TODO should prob aggregate these across all workers
                // TODO add a copySum method that allocates a pinned buffer, copies there then sumes into the wanted image.
                Image sum_ac, sum_cc;
                Image sumSq_ac, sumSq_cc;

                sum_ac.Allocate(montage_dim_x, montage_dim_y, 1);
                sumSq_ac.Allocate(montage_dim_x, montage_dim_y, 1);
                sum_cc.Allocate(montage_dim_x, montage_dim_y, 1);
                sumSq_cc.Allocate(montage_dim_x, montage_dim_y, 1);

                sum_ac.SetToConstant(0.0f);
                sumSq_ac.SetToConstant(0.0f);
                sum_cc.SetToConstant(0.0f);
                sumSq_cc.SetToConstant(0.0f);

                GPU[tIDX].d_sum3_ac.CopyDeviceToHost(sum_ac, true, false);
                GPU[tIDX].d_sumSq3_ac.CopyDeviceToHost(sumSq_ac, true, false);
                GPU[tIDX].d_sum3_cc.CopyDeviceToHost(sum_cc, true, false);
                GPU[tIDX].d_sumSq3_cc.CopyDeviceToHost(sumSq_cc, true, false);

                GPU[tIDX].d_max_intensity_projection_ac.Wait( );
                GPU[tIDX].d_max_intensity_projection_cc.Wait( );

                // TODO swap max_padding for explicit padding in x/y and limit calcs to that region.
                pixel_counter = 0;
                for ( current_y = 0; current_y < max_intensity_projection_ac.logical_y_dimension; current_y++ ) {
                    for ( current_x = 0; current_x < max_intensity_projection_ac.logical_x_dimension; current_x++ ) {
                        // first mip
                        if ( mip_buffer_ac.real_values[pixel_counter] > max_intensity_projection_ac.real_values[pixel_counter] ) {
                            max_intensity_projection_ac.real_values[pixel_counter] = mip_buffer_ac.real_values[pixel_counter];
                            best_psi_ac.real_values[pixel_counter]                 = psi_buffer_ac.real_values[pixel_counter];
                            best_theta_ac.real_values[pixel_counter]               = theta_buffer_ac.real_values[pixel_counter];
                            best_phi_ac.real_values[pixel_counter]                 = phi_buffer_ac.real_values[pixel_counter];
                        }

                        if ( mip_buffer_cc.real_values[pixel_counter] > max_intensity_projection_cc.real_values[pixel_counter] ) {
                            max_intensity_projection_cc.real_values[pixel_counter] = mip_buffer_cc.real_values[pixel_counter];
                            best_psi_cc.real_values[pixel_counter]                 = psi_buffer_cc.real_values[pixel_counter];
                            best_theta_cc.real_values[pixel_counter]               = theta_buffer_cc.real_values[pixel_counter];
                            best_phi_cc.real_values[pixel_counter]                 = phi_buffer_cc.real_values[pixel_counter];
                        }

                        correlation_pixel_sum_ac[pixel_counter] += (double)sum_ac.real_values[pixel_counter];
                        correlation_pixel_sum_of_squares_ac[pixel_counter] += (double)sumSq_ac.real_values[pixel_counter];
                        correlation_pixel_sum_cc[pixel_counter] += (double)sum_cc.real_values[pixel_counter];
                        correlation_pixel_sum_of_squares_cc[pixel_counter] += (double)sumSq_cc.real_values[pixel_counter];

                        pixel_counter++;
                    }
                    pixel_counter += max_intensity_projection_ac.padding_jump_value;
                }

                // GPU[tIDX].histogram.CopyToHostAndAdd(histogram_data);

                //                    current_correlation_position += GPU[tIDX].total_number_of_cccs_calculated;
                actual_number_of_ccs_calculated += GPU[tIDX].total_number_of_cccs_calculated;

            } // end of omp critical block
        } // end of parallel block

#endif
    }

    // post processing and saving outputs
    double sqrt_input_pixels = sqrt((double)(montage_image.logical_x_dimension * montage_image.logical_y_dimension));
    wxPrintf("N = %f\n", float(sqrt_input_pixels));
    max_intensity_projection_ac.MultiplyByConstant((float)sqrt_input_pixels);
    max_intensity_projection_cc.MultiplyByConstant((float)sqrt_input_pixels);
    max_intensity_projection_ac.QuickAndDirtyWriteSlice(mip_ac_filename.ToStdString( ), 1);
    max_intensity_projection_cc.QuickAndDirtyWriteSlice(mip_cc_filename.ToStdString( ), 1);

    for ( pixel_counter = 0; pixel_counter < montage_image.real_memory_allocated; pixel_counter++ ) {
        correlation_pixel_sum_ac_image.real_values[pixel_counter]            = (float)correlation_pixel_sum_ac[pixel_counter];
        correlation_pixel_sum_of_squares_ac_image.real_values[pixel_counter] = (float)correlation_pixel_sum_of_squares_ac[pixel_counter];
        correlation_pixel_sum_cc_image.real_values[pixel_counter]            = (float)correlation_pixel_sum_cc[pixel_counter];
        correlation_pixel_sum_of_squares_cc_image.real_values[pixel_counter] = (float)correlation_pixel_sum_of_squares_cc[pixel_counter];
    }

    for ( pixel_counter = 0; pixel_counter < montage_image.real_memory_allocated; pixel_counter++ ) {
        correlation_pixel_sum_ac[pixel_counter] /= float(total_correlation_positions_tm);
        correlation_pixel_sum_of_squares_ac[pixel_counter] = correlation_pixel_sum_of_squares_ac[pixel_counter] / float(total_correlation_positions_tm) - powf(correlation_pixel_sum_ac[pixel_counter], 2);
        correlation_pixel_sum_cc[pixel_counter] /= float(total_correlation_positions_tm);
        correlation_pixel_sum_of_squares_cc[pixel_counter] = correlation_pixel_sum_of_squares_cc[pixel_counter] / float(total_correlation_positions_tm) - powf(correlation_pixel_sum_cc[pixel_counter], 2);

        if ( correlation_pixel_sum_of_squares_ac[pixel_counter] > 0.0f ) {
            correlation_pixel_sum_of_squares_ac[pixel_counter] = sqrtf(correlation_pixel_sum_of_squares_ac[pixel_counter]) * (float)sqrt_input_pixels;
        }
        else
            correlation_pixel_sum_of_squares_ac[pixel_counter] = 0.0f;

        correlation_pixel_sum_ac[pixel_counter] *= (float)sqrt_input_pixels;

        if ( correlation_pixel_sum_of_squares_cc[pixel_counter] > 0.0f ) {
            correlation_pixel_sum_of_squares_cc[pixel_counter] = sqrtf(correlation_pixel_sum_of_squares_cc[pixel_counter]) * (float)sqrt_input_pixels;
        }
        else
            correlation_pixel_sum_of_squares_cc[pixel_counter] = 0.0f;

        correlation_pixel_sum_cc[pixel_counter] *= (float)sqrt_input_pixels;
    }

    // scaling

    for ( pixel_counter = 0; pixel_counter < max_intensity_projection_ac.real_memory_allocated; pixel_counter++ ) {
        max_intensity_projection_ac.real_values[pixel_counter] -= correlation_pixel_sum_ac[pixel_counter];
        max_intensity_projection_cc.real_values[pixel_counter] -= correlation_pixel_sum_cc[pixel_counter];

        if ( correlation_pixel_sum_of_squares_ac[pixel_counter] > 0.0f ) {
            max_intensity_projection_ac.real_values[pixel_counter] /= correlation_pixel_sum_of_squares_ac[pixel_counter];
        }
        else
            max_intensity_projection_ac.real_values[pixel_counter] = 0.0f;

        if ( correlation_pixel_sum_of_squares_cc[pixel_counter] > 0.0f ) {
            max_intensity_projection_cc.real_values[pixel_counter] /= correlation_pixel_sum_of_squares_cc[pixel_counter];
        }
        else
            max_intensity_projection_cc.real_values[pixel_counter] = 0.0f;
        correlation_pixel_sum_ac_image.real_values[pixel_counter]            = correlation_pixel_sum_ac[pixel_counter];
        correlation_pixel_sum_of_squares_ac_image.real_values[pixel_counter] = correlation_pixel_sum_of_squares_ac[pixel_counter];
        correlation_pixel_sum_cc_image.real_values[pixel_counter]            = correlation_pixel_sum_cc[pixel_counter];
        correlation_pixel_sum_of_squares_cc_image.real_values[pixel_counter] = correlation_pixel_sum_of_squares_cc[pixel_counter];
    }

    max_intensity_projection_ac.QuickAndDirtyWriteSlice(scaled_mip_ac_filename.ToStdString( ), 1);
    max_intensity_projection_cc.QuickAndDirtyWriteSlice(scaled_mip_cc_filename.ToStdString( ), 1);
    correlation_pixel_sum_ac_image.QuickAndDirtyWriteSlice(avg_ac_filename.ToStdString( ), 1);
    correlation_pixel_sum_cc_image.QuickAndDirtyWriteSlice(avg_cc_filename.ToStdString( ), 1);

    best_psi_ac.QuickAndDirtyWriteSlice(best_psi_ac_filename.ToStdString( ), 1);
    best_psi_cc.QuickAndDirtyWriteSlice(best_psi_cc_filename.ToStdString( ), 1);
    best_theta_ac.QuickAndDirtyWriteSlice(best_theta_ac_filename.ToStdString( ), 1);
    best_theta_cc.QuickAndDirtyWriteSlice(best_theta_cc_filename.ToStdString( ), 1);
    best_phi_ac.QuickAndDirtyWriteSlice(best_phi_ac_filename.ToStdString( ), 1);
    best_phi_cc.QuickAndDirtyWriteSlice(best_phi_cc_filename.ToStdString( ), 1);

    correlation_pixel_sum_of_squares_ac_image.QuickAndDirtyWriteSlice(std_ac_filename.ToStdString( ), 1);
    correlation_pixel_sum_of_squares_cc_image.QuickAndDirtyWriteSlice(std_cc_filename.ToStdString( ), 1);
#ifdef ENABLEGPU
    if ( use_gpu ) {
        delete[] GPU;
    }

#endif

    wxPrintf("\n\n\tTimings: Overall: %s\n", (wxDateTime::Now( ) - overall_start).Format( ));

    return true;
}

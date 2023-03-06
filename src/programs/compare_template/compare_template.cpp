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
    // wxString log_output_file;
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
    float    high_resolution_limit   = 8.0;
    float    angular_step            = 5.0;
    int      best_parameters_to_keep = 20;
    float    padding                 = 1.0;
    wxString my_symmetry             = "C1";
    float    in_plane_angular_step   = 0;
    int      max_threads             = 1;
    bool     use_gpu_input           = false;
    int      number_of_sampled_views = 30;
    int      result_number;
    bool     use_existing_params = false;
    wxString preexisting_particle_file_name;
    int      n_Frames_for_simulating_shot_noise = 0;

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
    padding                                = my_input->GetFloatFromUser("Padding factor", "Factor determining how much the input volume is padded to improve projections", "1.0", 1.0, 2.0);
    my_symmetry                            = my_input->GetSymmetryFromUser("Template symmetry", "The symmetry of the template reconstruction", "C1");
    angular_step                           = my_input->GetFloatFromUser("TM out of plane angular step (0.0 = set automatically)", "Angular step size for global grid search", "0.0", 0.0);
    in_plane_angular_step                  = my_input->GetFloatFromUser("TM in plane angular step (0.0 = set automatically)", "Angular step size for in-plane rotations during the search", "0.0", 0.0);
    //log_output_file                        = my_input->GetFilenameFromUser("Log file for recording meta data", "Log output file", "log.txt", false);
#ifdef ENABLEGPU
    use_gpu_input = my_input->GetYesNoFromUser("Use GPU", "Offload expensive calcs to GPU", "No");
    max_threads   = my_input->GetIntFromUser("Max. threads to use for calculation", "when threading, what is the max threads to run", "1", 1);
#endif
    use_existing_params = my_input->GetYesNoFromUser("Use an existing set of orientations", "yes no", "no");
    if ( ! use_existing_params )
        number_of_sampled_views = my_input->GetIntFromUser("Number of sampled views", "number of sampled views", "1", 1, 10000); //TODO FIXME
    else {
        preexisting_particle_file_name = my_input->GetFilenameFromUser("cisTEM star file name", "an input star file to match reconstruction", "myparams.star", true);
    }
    scaled_mip_ac_filename             = my_input->GetFilenameFromUser("Output scaled mip image ac", "Output scaled mip image ac", "scaled_mip_ac.mrc", false);
    scaled_mip_cc_filename             = my_input->GetFilenameFromUser("Output scaled mip image cc", "Output scaled mip image ac", "scaled_mip_cc.mrc", false);
    mip_ac_filename                    = my_input->GetFilenameFromUser("Output mip image ac", "Output mip image ac", "mip_ac.mrc", false);
    mip_cc_filename                    = my_input->GetFilenameFromUser("Output mip image cc", "Output mip image cc", "mip_cc.mrc", false);
    avg_ac_filename                    = my_input->GetFilenameFromUser("Output avg image ac", "Output avg image ac", "avg_ac.mrc", false);
    avg_cc_filename                    = my_input->GetFilenameFromUser("Output avg image cc", "Output avg image cc", "avg_cc.mrc", false);
    std_ac_filename                    = my_input->GetFilenameFromUser("Output std image ac", "Output std image ac", "std_cc.mrc", false);
    std_cc_filename                    = my_input->GetFilenameFromUser("Output std image cc", "Output std image cc", "std_cc.mrc", false);
    best_psi_ac_filename               = my_input->GetFilenameFromUser("Output autocorrelation psi file", "The file for saving the best psi image", "psi.mrc", false);
    best_theta_ac_filename             = my_input->GetFilenameFromUser("Output autocorrelation theta file", "The file for saving the best theta image", "theta.mrc", false);
    best_phi_ac_filename               = my_input->GetFilenameFromUser("Output autocorrelation phi file", "The file for saving the best phi image", "phi.mrc", false);
    best_psi_cc_filename               = my_input->GetFilenameFromUser("Output cross-correlation psi file", "The file for saving the best psi image", "psi.mrc", false);
    best_theta_cc_filename             = my_input->GetFilenameFromUser("Output cross-correlation theta file", "The file for saving the best theta image", "theta.mrc", false);
    best_phi_cc_filename               = my_input->GetFilenameFromUser("Output cross-correlation phi file", "The file for saving the best phi image", "phi.mrc", false);
    data_directory_name                = my_input->GetFilenameFromUser("Name for data directory", "path to data directory", "60_120_5_2.5", false);
    result_number                      = my_input->GetIntFromUser("Result number", "result number", "1", 1, 400);
    n_Frames_for_simulating_shot_noise = my_input->GetIntFromUser("Frames for simulating shot noise (0 is no shot noise)", "shot noise", "0", 0, 100);

    int first_search_position = -1;
    int last_search_position  = -1;

    delete my_input;

    my_current_job.ManualSetArguments("tttttffffffffifftiffitttttttttttttttbibitii",
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
                                      best_parameters_to_keep, //i
                                      padding,
                                      phase_shift,
                                      my_symmetry.ToUTF8( ).data( ),
                                      first_search_position,
                                      angular_step,
                                      in_plane_angular_step,
                                      last_search_position,
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
                                      use_gpu_input,
                                      max_threads,
                                      use_existing_params,
                                      number_of_sampled_views,
                                      preexisting_particle_file_name.ToUTF8( ).data( ),
                                      result_number,
                                      n_Frames_for_simulating_shot_noise);
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
    float    high_resolution_limit_search           = my_current_job.arguments[12].ReturnFloatArgument( );
    int      best_parameters_to_keep                = my_current_job.arguments[13].ReturnIntegerArgument( );
    float    padding                                = my_current_job.arguments[14].ReturnFloatArgument( );
    float    phase_shift                            = my_current_job.arguments[15].ReturnFloatArgument( );
    wxString my_symmetry                            = my_current_job.arguments[16].ReturnStringArgument( );
    int      first_search_position                  = my_current_job.arguments[17].ReturnIntegerArgument( );
    float    angular_step                           = my_current_job.arguments[18].ReturnFloatArgument( );
    float    in_plane_angular_step                  = my_current_job.arguments[19].ReturnFloatArgument( );
    int      last_search_position                   = my_current_job.arguments[20].ReturnIntegerArgument( );
    // wxString log_output_file                        = my_current_job.arguments[24].ReturnStringArgument( );
    wxString scaled_mip_ac_filename             = my_current_job.arguments[21].ReturnStringArgument( );
    wxString scaled_mip_cc_filename             = my_current_job.arguments[22].ReturnStringArgument( );
    wxString mip_ac_filename                    = my_current_job.arguments[23].ReturnStringArgument( );
    wxString mip_cc_filename                    = my_current_job.arguments[24].ReturnStringArgument( );
    wxString avg_ac_filename                    = my_current_job.arguments[25].ReturnStringArgument( );
    wxString avg_cc_filename                    = my_current_job.arguments[26].ReturnStringArgument( );
    wxString std_ac_filename                    = my_current_job.arguments[27].ReturnStringArgument( );
    wxString std_cc_filename                    = my_current_job.arguments[28].ReturnStringArgument( );
    wxString best_psi_ac_filename               = my_current_job.arguments[29].ReturnStringArgument( );
    wxString best_theta_ac_filename             = my_current_job.arguments[30].ReturnStringArgument( );
    wxString best_phi_ac_filename               = my_current_job.arguments[31].ReturnStringArgument( );
    wxString best_psi_cc_filename               = my_current_job.arguments[32].ReturnStringArgument( );
    wxString best_theta_cc_filename             = my_current_job.arguments[33].ReturnStringArgument( );
    wxString best_phi_cc_filename               = my_current_job.arguments[34].ReturnStringArgument( );
    wxString data_directory_name                = my_current_job.arguments[35].ReturnStringArgument( );
    bool     use_gpu                            = my_current_job.arguments[36].ReturnBoolArgument( );
    int      max_threads                        = my_current_job.arguments[37].ReturnIntegerArgument( );
    bool     use_existing_params                = my_current_job.arguments[38].ReturnBoolArgument( );
    int      number_of_sampled_views            = my_current_job.arguments[39].ReturnIntegerArgument( );
    wxString preexisting_particle_file_name     = my_current_job.arguments[40].ReturnStringArgument( );
    int      result_number                      = my_current_job.arguments[41].ReturnIntegerArgument( );
    int      n_Frames_for_simulating_shot_noise = my_current_job.arguments[42].ReturnIntegerArgument( );

    // This condition applies to GUI and CLI - it is just a recommendation to the user.
    if ( use_gpu && max_threads <= 1 ) {
        SendInfo("Warning, you are only using one thread on the GPU. Suggested minimum is 2. Check compute saturation using nvidia-smi -l 1\n");
    }
    if ( ! use_gpu ) {
        SendInfo("GPU disabled\nCan use up to 44 threads on roma\n.");
    }

    int  padded_dimensions_x;
    int  padded_dimensions_y;
    int  pad_factor          = 6;
    int  number_of_rotations = 0;
    long total_correlation_positions;
    long current_correlation_position_sampled_view, current_correlation_position; //TODO
    long total_correlation_positions_per_thread;
    long pixel_counter;

    int   current_search_position;
    float current_psi;
    float psi_step  = in_plane_angular_step;
    float psi_max   = 360.0f;
    float psi_start = 0.0f;

    ParameterMap parameter_map; // needed for euler search init
    parameter_map.SetAllTrue( );
    float variance;

    Curve whitening_filter;
    Curve number_of_terms;
    // NumericTextFile log_file(log_output_file, OPEN_TO_WRITE, 1);

    float* psi_tm;
    float* theta_tm;
    float* phi_tm;

    cisTEMParameters input_star_file;
    long             number_preexisting_particles;
    cisTEMParameters output_star_file;

    Image           input_reconstruction_particle, input_reconstruction_correct, input_reconstruction_wrong;
    ImageFile       input_search_image_file;
    ImageFile       input_reconstruction_particle_file, input_reconstruction_correct_template_file, input_reconstruction_wrong_template_file;
    Image           input_image;
    Image           montage_image, montage_image_stack;
    EulerSearch     global_euler_search;
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
    global_euler_search.InitGrid(my_symmetry, angular_step, 0.0f, 0.0f, psi_max, psi_step, psi_start, pixel_size / high_resolution_limit_search, parameter_map, best_parameters_to_keep);
    if ( my_symmetry.StartsWith("C") ) // TODO 2x check me - w/o this O symm at least is broken
    {
        if ( global_euler_search.test_mirror == true ) // otherwise the theta max is set to 90.0 and test_mirror is set to true.  However, I don't want to have to test the mirrors.
        {
            global_euler_search.theta_max = 180.0f;
        }
    }
    global_euler_search.CalculateGridSearchPositions(false);

    first_search_position = 0;
    last_search_position  = global_euler_search.number_of_search_positions - 1;

    total_correlation_positions               = 0;
    current_correlation_position_sampled_view = 0;
    current_correlation_position              = 0;

    if ( use_existing_params ) {
        if ( DoesFileExist(preexisting_particle_file_name) ) {
            input_star_file.ReadFromcisTEMStarFile(preexisting_particle_file_name);
            number_preexisting_particles = input_star_file.ReturnNumberofLines( );
            wxPrintf("\nFound %ld particles in the input star file\n", number_preexisting_particles);
        }
        else {
            SendErrorAndCrash(wxString::Format("Error: Input star file %s not found\n", preexisting_particle_file_name));
        }
        number_of_sampled_views = number_preexisting_particles;
    }

    for ( current_search_position = first_search_position; current_search_position <= last_search_position; current_search_position++ ) {
        //loop over each rotation angle

        for ( current_psi = psi_start; current_psi <= psi_max; current_psi += psi_step ) {
            total_correlation_positions++;
        }
    }

    for ( current_psi = psi_start; current_psi <= psi_max; current_psi += psi_step ) {
        number_of_rotations++;
    }

    wxPrintf("TM step sizes:\n");
    wxPrintf("Searching %i positions on the Euler sphere (first-last: %i-%i)\n", last_search_position - first_search_position, first_search_position, last_search_position);
    wxPrintf("Searching %i rotations per position.\n", number_of_rotations);
    wxPrintf("There are %li correlation positions total.\n\n", total_correlation_positions);

    psi_tm   = new float[number_of_rotations];
    theta_tm = new float[last_search_position - first_search_position + 1];
    phi_tm   = new float[last_search_position - first_search_position + 1];

    for ( int k = 0; k < number_of_rotations; k++ ) {
        psi_tm[k] = psi_start + psi_step * k;
        //   wxPrintf("psi = %f\n", psi_tm[k]);
    }
    for ( int k = first_search_position; k <= last_search_position; k++ ) {
        phi_tm[k]   = global_euler_search.list_of_search_parameters[k][0];
        theta_tm[k] = global_euler_search.list_of_search_parameters[k][1];
        //   wxPrintf("theta phi = %f %f\n", phi_tm[k], theta_tm[k]);
    }

    output_star_file.PreallocateMemoryAndBlank(number_of_sampled_views);
    output_star_file.parameters_to_write.SetAllToTrue( );
    output_star_file.parameters_to_write.image_is_active         = false;
    output_star_file.parameters_to_write.original_image_filename = false;
    output_star_file.parameters_to_write.reference_3d_filename   = false;
    output_star_file.parameters_to_write.stack_filename          = false;

    cisTEMParameterLine parameters;
    // set default parameter values
    parameters.position_in_stack                  = 0;
    parameters.image_is_active                    = 0;
    parameters.psi                                = 0.0f;
    parameters.theta                              = 0.0f;
    parameters.phi                                = 0.0f;
    parameters.x_shift                            = 0.0f;
    parameters.y_shift                            = 0.0f;
    parameters.defocus_1                          = defocus1;
    parameters.defocus_2                          = defocus2;
    parameters.defocus_angle                      = defocus_angle;
    parameters.phase_shift                        = phase_shift;
    parameters.occupancy                          = 100.0f;
    parameters.logp                               = -1000.0f;
    parameters.sigma                              = 10.0f;
    parameters.score                              = 10.0f;
    parameters.score_change                       = 0.0f;
    parameters.pixel_size                         = pixel_size;
    parameters.microscope_voltage_kv              = voltage_kV;
    parameters.microscope_spherical_aberration_mm = spherical_aberration_mm;
    parameters.amplitude_contrast                 = 0.0f;
    parameters.beam_tilt_x                        = 0.0f;
    parameters.beam_tilt_y                        = 0.0f;
    parameters.image_shift_x                      = 0.0f;
    parameters.image_shift_y                      = 0.0f;
    parameters.stack_filename                     = output_pose_filename;
    parameters.original_image_filename            = wxEmptyString;
    parameters.reference_3d_filename              = wxEmptyString;
    parameters.best_2d_class                      = 0;
    parameters.beam_tilt_group                    = 0;
    parameters.particle_group                     = 0;
    parameters.pre_exposure                       = 0.0f;
    parameters.total_exposure                     = 0.0f;

    // These vars are only needed in the GPU code, but also need to be set out here to compile.
    // update GPU setup to "inner" loop only - for tm search
    bool first_gpu_loop = true;
    int  nGPUs          = 1;
    int  nJobs          = last_search_position - first_search_position + 1;
    if ( use_gpu && max_threads > nJobs ) {
        wxPrintf("\n\tWarning, you request more threads (%d) than there are search positions (%d)\n", max_threads, nJobs);
        max_threads = nJobs;
    }

    int minPos = first_search_position;
    int maxPos = last_search_position;
    int incPos = (nJobs) / (max_threads);

#ifdef ENABLEGPU
    TemplateSnrRatioCore* GPU;
    DeviceManager         gpuDev;
#endif

    if ( use_gpu ) {
        total_correlation_positions_per_thread = total_correlation_positions / max_threads;

#ifdef ENABLEGPU
        //    checkCudaErrors(cudaGetDeviceCount(&nGPUs));
        GPU = new TemplateSnrRatioCore[max_threads];
        gpuDev.Init(nGPUs);

#endif
    }

    ProgressBar* my_progress;
    my_progress = new ProgressBar(total_correlation_positions_per_thread);

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

    // prepare stack of sampled views
    montage_image_stack.Allocate(input_reconstruction_particle_file.ReturnXSize( ), input_reconstruction_particle_file.ReturnXSize( ), number_of_sampled_views);
    montage_image_stack.SetToConstant(0.0f);

    int   j = 0, view_counter = 0;
    int   current_x, current_y;
    Image current_projection_image, current_projection_other, current_projection_correct_template, padded_projection_image;
    float ac_val, ac_max, cc_val, cc_max;

    // generate particle montage
    current_projection_image.Allocate(input_reconstruction_particle_file.ReturnXSize( ), input_reconstruction_particle_file.ReturnXSize( ), 1, false);
    if ( padding != 1.0f )
        padded_projection_image.Allocate(input_reconstruction_particle_file.ReturnXSize( ) * padding, input_reconstruction_particle_file.ReturnXSize( ) * padding, false);

    // sample views

    float* sampled_psi;
    float* sampled_theta;
    float* sampled_phi;
    sampled_psi   = new float[number_of_sampled_views];
    sampled_theta = new float[number_of_sampled_views];
    sampled_phi   = new float[number_of_sampled_views];
    for ( int iView = 0; iView < number_of_sampled_views; iView++ ) {
        if ( use_existing_params ) {
            sampled_psi[iView]   = input_star_file.ReturnPsi(iView);
            sampled_theta[iView] = input_star_file.ReturnTheta(iView);
            sampled_phi[iView]   = input_star_file.ReturnPhi(iView);
            wxPrintf("psi theta phi = %f %f %f \n", sampled_psi[iView], sampled_theta[iView], sampled_phi[iView]);
        }
        else {
            int k1               = rand( ) % number_of_rotations;
            int k2               = rand( ) % (last_search_position - first_search_position + 1);
            sampled_psi[iView]   = psi_tm[k1];
            sampled_theta[iView] = theta_tm[k2];
            sampled_phi[iView]   = phi_tm[k1];
            wxPrintf("k1 = %i k2=%i\n", k1, k2);
            wxPrintf("psi theta phi = %f %f %f \n", sampled_psi[iView], sampled_theta[iView], sampled_phi[iView]);
        }

        // generate particle from sampled view (in TM we applied projection filter first then normalized the templates)
        angles_sampled_view.Init(sampled_phi[iView], sampled_theta[iView], sampled_psi[iView], 0.0, 0.0);
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
        if ( n_Frames_for_simulating_shot_noise == 0 )
            montage_image_stack.InsertOtherImageAtSpecifiedSlice(&current_projection_image, iView);
        else {
            float min_value, max_value;
            current_projection_image.GetMinMax(min_value, max_value);
            // simulate shot noise //can we determing the nFrames based on SNR from simulator
            Image output_image;
            output_image.Allocate(current_projection_image.logical_x_dimension, current_projection_image.logical_y_dimension, 1);
            output_image.SetToConstant(0.0f);

            Image temp_image;
            temp_image.Allocate(current_projection_image.logical_x_dimension, current_projection_image.logical_y_dimension, 1);
            current_projection_image.AddConstant(-min_value);
            current_projection_image.DivideByConstant(max_value - min_value); //perform min-max scaling
            for ( int iFrame = 0; iFrame < n_Frames_for_simulating_shot_noise; iFrame++ ) {
                temp_image.SetToConstant(0.0f);
                RandomNumberGenerator my_rand(PIf);
                for ( long iPixel = 0; iPixel < current_projection_image.real_memory_allocated; iPixel++ ) {
                    temp_image.real_values[iPixel] = my_rand.GetNormalRandomSTD(current_projection_image.real_values[iPixel], sqrtf(current_projection_image.real_values[iPixel])); //distribution(gen);
                }
                output_image.AddImage(&temp_image);
            }
            output_image.ForwardFFT( );

            output_image.ZeroCentralPixel( ); // equivalent to subtracting mean in real space
            output_image.DivideByConstant(sqrtf(output_image.ReturnSumOfSquares( ))); //todo: is scaling necessary
            output_image.BackwardFFT( );

            montage_image_stack.InsertOtherImageAtSpecifiedSlice(&output_image, iView);
        }

        // update parameters
        parameters.position_in_stack = iView + 1;
        parameters.psi               = sampled_psi[iView];
        parameters.theta             = sampled_theta[iView];
        parameters.phi               = sampled_phi[iView];
        output_star_file.all_parameters.Add(parameters);
    }
    output_star_file.WriteTocisTEMStarFile(output_pose_filename.ToStdString( ));
    montage_image_stack.QuickAndDirtyWriteSlices(wxString::Format("%s/montage_stack_%.1f_%.1f_%i_%i.mrc", data_directory_name, psi_step, angular_step, number_of_sampled_views, result_number).ToStdString( ), 1, number_of_sampled_views);
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
    montage_stack_file.OpenFile(wxString::Format("%s/montage_stack_%.1f_%.1f_%i_%i.mrc", data_directory_name, psi_step, angular_step, number_of_sampled_views, result_number).ToStdString( ), false);
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
    montage_image.QuickAndDirtyWriteSlice(wxString::Format("%s/montage_%.1f_%.1f_%i_%i.mrc", data_directory_name, psi_step, angular_step, number_of_sampled_views, result_number).ToStdString( ), 1);

    // normalize stitched image

    // montage_image.ReplaceOutliersWithMean(5.0f);
    montage_image.ForwardFFT( );
    montage_image.SwapRealSpaceQuadrants( );
    montage_image.ZeroCentralPixel( ); // equivalent to subtracting mean in real space
    montage_image.DivideByConstant(sqrtf(montage_image.ReturnSumOfSquares( )));

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

                GPU[tIDX].Init(this, input_reconstruction_correct, input_reconstruction_wrong, montage_image, current_projection_correct_template, current_projection_other, psi_max, psi_start, psi_step, angles_tm, global_euler_search, t_first_search_position, t_last_search_position, my_progress, number_of_rotations, total_correlation_positions, total_correlation_positions_per_thread, data_directory_name);

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

            GPU[tIDX].RunInnerLoop(projection_filter, tIDX, current_correlation_position); // current_correlation_position_sampled_view?

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
        correlation_pixel_sum_ac[pixel_counter] /= float(total_correlation_positions);
        correlation_pixel_sum_of_squares_ac[pixel_counter] = correlation_pixel_sum_of_squares_ac[pixel_counter] / float(total_correlation_positions) - powf(correlation_pixel_sum_ac[pixel_counter], 2);
        correlation_pixel_sum_cc[pixel_counter] /= float(total_correlation_positions);
        correlation_pixel_sum_of_squares_cc[pixel_counter] = correlation_pixel_sum_of_squares_cc[pixel_counter] / float(total_correlation_positions) - powf(correlation_pixel_sum_cc[pixel_counter], 2);

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

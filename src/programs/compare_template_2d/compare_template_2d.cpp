#include "../../core/core_headers.h"
#include "../../core/cistem_constants.h"

class
        CompareTemplate2DApp : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(CompareTemplate2DApp)

// override the DoInteractiveUserInput

void CompareTemplate2DApp::DoInteractiveUserInput( ) {
    wxString input_search_images;
    wxString input_reconstruction_particle_filename, input_reconstruction_correct_filename, input_reconstruction_wrong_filename;
    // wxString log_output_file;
    wxString output_pose_filename;

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
    bool     use_existing_params  = false;
    bool     use_existing_defocus = false;
    wxString preexisting_particle_file_name;

    UserInput* my_input = new UserInput("CompareTemplate2D", 1.00);

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
    max_threads         = my_input->GetIntFromUser("Max. threads to use for calculation", "when threading, what is the max threads to run", "1", 1);
    use_existing_params = my_input->GetYesNoFromUser("Use an existing set of orientations", "yes no", "no");
    if ( ! use_existing_params )
        number_of_sampled_views = my_input->GetIntFromUser("Number of sampled views", "number of sampled views", "1", 1, 10000);
    else {
        preexisting_particle_file_name = my_input->GetFilenameFromUser("cisTEM star file name", "an input star file to match reconstruction", "myparams.star", true);
        use_existing_defocus           = my_input->GetYesNoFromUser("Use defocus from input star file", "yes no", "no");
    }
    result_number = my_input->GetIntFromUser("Result number", "result number", "1", 1, 400);

    int first_search_position = -1;
    int last_search_position  = -1;

    delete my_input;

    my_current_job.ManualSetArguments("tttttffffffffifftiffiibbiti",
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
                                      best_parameters_to_keep,
                                      padding,
                                      phase_shift,
                                      my_symmetry.ToUTF8( ).data( ),
                                      first_search_position,
                                      angular_step,
                                      in_plane_angular_step,
                                      last_search_position,
                                      max_threads,
                                      use_existing_params,
                                      use_existing_defocus,
                                      number_of_sampled_views,
                                      preexisting_particle_file_name.ToUTF8( ).data( ),
                                      result_number);
}

// override the do calculation method which will be what is actually run..

bool CompareTemplate2DApp::DoCalculation( ) {
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
    int      max_threads                    = my_current_job.arguments[21].ReturnIntegerArgument( );
    bool     use_existing_params            = my_current_job.arguments[22].ReturnBoolArgument( );
    bool     use_existing_defocus           = my_current_job.arguments[23].ReturnBoolArgument( );
    int      number_of_sampled_views        = my_current_job.arguments[24].ReturnIntegerArgument( );
    wxString preexisting_particle_file_name = my_current_job.arguments[25].ReturnStringArgument( );
    int      result_number                  = my_current_job.arguments[26].ReturnIntegerArgument( );

    int  number_of_rotations = 0;
    long total_correlation_positions;
    long total_correlation_positions_per_thread;
    long pixel_counter;

    int   current_search_position;
    float current_psi;
    float psi_step  = in_plane_angular_step;
    float psi_max   = 360.0f;
    float psi_start = 0.0f;

    ParameterMap parameter_map; // needed for euler search init
    parameter_map.SetAllTrue( );

    Curve whitening_filter;
    Curve number_of_terms;
    // NumericTextFile log_file(log_output_file, OPEN_TO_WRITE, 1);

    cisTEMParameters output_star_file;

    Image           input_reconstruction_particle, input_reconstruction_correct, input_reconstruction_wrong;
    ImageFile       input_search_image_file;
    ImageFile       input_reconstruction_particle_file, input_reconstruction_correct_template_file, input_reconstruction_wrong_template_file;
    Image           input_image;
    Image           montage_image, montage_image_stack;
    EulerSearch     global_euler_search;
    AnglesAndShifts angles;

    input_search_image_file.OpenFile(input_search_images_filename.ToStdString( ), false);
    input_image.ReadSlice(&input_search_image_file, 1);

    input_reconstruction_particle_file.OpenFile(input_reconstruction_particle_filename.ToStdString( ), false);
    input_reconstruction_correct_template_file.OpenFile(input_reconstruction_correct_filename.ToStdString( ), false);
    input_reconstruction_wrong_template_file.OpenFile(input_reconstruction_wrong_filename.ToStdString( ), false);

    input_reconstruction_particle.ReadSlices(&input_reconstruction_particle_file, 1, input_reconstruction_particle_file.ReturnNumberOfSlices( )); // particle in image
    input_reconstruction_correct.ReadSlices(&input_reconstruction_correct_template_file, 1, input_reconstruction_correct_template_file.ReturnNumberOfSlices( )); // correct template
    input_reconstruction_wrong.ReadSlices(&input_reconstruction_wrong_template_file, 1, input_reconstruction_wrong_template_file.ReturnNumberOfSlices( )); // wrong template

    // TODO 1. normalize (done) 2. pad image with mean and pad template with 0 to same size 3. pad template to remove aliasing (done)
    cisTEMParameters input_star_file;
    long             number_preexisting_particles;
    float*           psi_tm;
    float*           theta_tm;
    float*           phi_tm;
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
    else {
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

        total_correlation_positions = 0;

        for ( current_search_position = first_search_position; current_search_position <= last_search_position; current_search_position++ ) {
            //loop over each rotation angle

            for ( current_psi = psi_start; current_psi <= psi_max; current_psi += psi_step ) {
                total_correlation_positions++;
            }
        }

        for ( current_psi = psi_start; current_psi <= psi_max; current_psi += psi_step ) {
            number_of_rotations++;
        }

        wxPrintf("Sampling from:\n");
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
    }
    wxPrintf("number of sampled views=%i\n", number_of_sampled_views);

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
    if ( ! use_existing_params || (use_existing_params && ! use_existing_defocus) ) {
        input_ctf.SetDefocus(defocus1 / pixel_size, defocus2 / pixel_size, deg_2_rad(defocus_angle));
        projection_filter.CalculateCTFImage(input_ctf);
        projection_filter.ApplyCurveFilter(&whitening_filter);
    }

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

    int   j = 0, view_counter = 0;
    Image current_projection_image, current_projection_other, current_projection_correct_template, padded_projection_image;

    //NumericTextFile pose_file(output_pose_filename, OPEN_TO_WRITE, 3);
    //pose_file.WriteCommentLine("Psi, Theta, Phi");
    //double temp_double_array[3];

    current_projection_image.Allocate(input_reconstruction_particle_file.ReturnXSize( ), input_reconstruction_particle_file.ReturnXSize( ), 1, false);
    if ( padding != 1.0f )
        padded_projection_image.Allocate(input_reconstruction_particle_file.ReturnXSize( ) * padding, input_reconstruction_particle_file.ReturnXSize( ) * padding, false);

    // for loop: generate particle and calculate AC/CC
    float sum_ac   = 0;
    float sum_cc   = 0;
    float sos_ac   = 0;
    float sos_cc   = 0;
    float ratio    = 0;
    float ratio_sq = 0;

    current_projection_other.Allocate(input_reconstruction_particle_file.ReturnXSize( ), input_reconstruction_particle_file.ReturnXSize( ), 1, false);
    current_projection_correct_template.Allocate(input_reconstruction_particle_file.ReturnXSize( ), input_reconstruction_particle_file.ReturnXSize( ), 1, false);

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
            if ( use_existing_defocus ) {
                defocus1      = input_star_file.ReturnDefocus1(iView);
                defocus2      = input_star_file.ReturnDefocus2(iView);
                defocus_angle = input_star_file.ReturnDefocusAngle(iView);
                input_ctf.SetDefocus(defocus1 / pixel_size, defocus2 / pixel_size, deg_2_rad(defocus_angle));
                projection_filter.CalculateCTFImage(input_ctf);
                projection_filter.ApplyCurveFilter(&whitening_filter);
            }
            wxPrintf("psi theta phi defocus1 defocus2 defocus_angle= %f %f %f %f %f %f\n", sampled_psi[iView], sampled_theta[iView], sampled_phi[iView], defocus1, defocus2, defocus_angle);
        }
        else {
            int k1               = rand( ) % number_of_rotations;
            int k2               = rand( ) % (last_search_position - first_search_position + 1);
            sampled_psi[iView]   = psi_tm[k1];
            sampled_theta[iView] = theta_tm[k2];
            sampled_phi[iView]   = phi_tm[k1];
            wxPrintf("k1 = %i k2=%i\n", k1, k2);
            wxPrintf("psi theta phi defocus1 defocus2 defocus_angle= %f %f %f %f %f %f\n", sampled_psi[iView], sampled_theta[iView], sampled_phi[iView], defocus1, defocus2, defocus_angle);
        }

        angles.Init(sampled_phi[iView], sampled_theta[iView], sampled_psi[iView], 0.0, 0.0);
        //angles.Init(144.000000, 60.000000, 180.000000, 0, 0);
        // generate particle from sampled view (in TM we applied projection filter first then normalized the templates)
        if ( padding != 1.0f ) {
            input_reconstruction_particle.ExtractSlice(padded_projection_image, angles, 1.0f, false);
            padded_projection_image.SwapRealSpaceQuadrants( );
            padded_projection_image.BackwardFFT( );
            padded_projection_image.ClipInto(&current_projection_image);
            current_projection_image.ForwardFFT( );
        }
        else {
            input_reconstruction_particle.ExtractSlice(current_projection_image, angles, 1.0f, false);
            current_projection_image.SwapRealSpaceQuadrants( );
        }
        current_projection_image.MultiplyPixelWise(projection_filter);
        current_projection_image.ZeroCentralPixel( );
        current_projection_image.DivideByConstant(sqrtf(current_projection_image.ReturnSumOfSquares( )));

        input_reconstruction_correct.ExtractSlice(current_projection_correct_template, angles, 1.0f, false);
        input_reconstruction_wrong.ExtractSlice(current_projection_other, angles, 1.0f, false);

        current_projection_correct_template.SwapRealSpaceQuadrants( );
        current_projection_other.SwapRealSpaceQuadrants( );

        current_projection_correct_template.MultiplyPixelWise(projection_filter);
        current_projection_other.MultiplyPixelWise(projection_filter);

        current_projection_correct_template.BackwardFFT( );
        current_projection_other.BackwardFFT( );

        current_projection_correct_template.AddConstant(-current_projection_correct_template.ReturnAverageOfRealValuesOnEdges( ));
        current_projection_other.AddConstant(current_projection_other.ReturnAverageOfRealValuesOnEdges( ));

        current_projection_correct_template.DivideByConstant(sqrtf(current_projection_correct_template.ReturnVarianceOfRealValues( )));
        current_projection_other.DivideByConstant(sqrtf(current_projection_other.ReturnVarianceOfRealValues( )));

        current_projection_correct_template.ForwardFFT( );
        current_projection_correct_template.ZeroCentralPixel( );

        current_projection_other.ForwardFFT( );
        current_projection_other.ZeroCentralPixel( );

#ifdef MKL
        // Use the MKL
        vmcMulByConj(current_projection_correct_template.real_memory_allocated / 2, reinterpret_cast<MKL_Complex8*>(current_projection_image.complex_values), reinterpret_cast<MKL_Complex8*>(current_projection_correct_template.complex_values), reinterpret_cast<MKL_Complex8*>(current_projection_correct_template.complex_values), VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
#else
        for ( pixel_counter = 0; pixel_counter < current_projection_correct_template.real_memory_allocated / 2; pixel_counter++ ) {
            current_projection_correct_template.complex_values[pixel_counter] = conj(current_projection_correct_template.complex_values[pixel_counter]) * current_projection_image.complex_values[pixel_counter];
        }
#endif

#ifdef MKL
        // Use the MKL
        vmcMulByConj(current_projection_other.real_memory_allocated / 2, reinterpret_cast<MKL_Complex8*>(current_projection_image.complex_values), reinterpret_cast<MKL_Complex8*>(current_projection_other.complex_values), reinterpret_cast<MKL_Complex8*>(current_projection_other.complex_values), VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
#else
        for ( pixel_counter = 0; pixel_counter < current_projection_other.real_memory_allocated / 2; pixel_counter++ ) {
            current_projection_other.complex_values[pixel_counter] = conj(current_projection_other.complex_values[pixel_counter]) * current_projection_image.complex_values[pixel_counter];
        }
#endif

        current_projection_correct_template.BackwardFFT( );
        current_projection_other.BackwardFFT( );

        float ac = current_projection_correct_template.ReturnMaximumValue( );
        float cc = current_projection_other.ReturnMaximumValue( );

        //sum_ac += ac;
        //sum_cc += cc;
        //sos_ac += pow(ac, 2);
        //sos_cc += pow(cc, 2);
        ratio += log2f(ac / cc);
        ratio_sq += pow(log2f(ac / cc), 2);

        // update parameters
        parameters.position_in_stack = iView + 1;
        parameters.psi               = sampled_psi[iView];
        parameters.theta             = sampled_theta[iView];
        parameters.phi               = sampled_phi[iView];
        output_star_file.all_parameters.Add(parameters);
    }
    output_star_file.WriteTocisTEMStarFile(output_pose_filename.ToStdString( ));

    ratio /= number_of_sampled_views;
    ratio_sq /= number_of_sampled_views;

    float variance = ratio_sq - pow(ratio, 2);
    wxPrintf("mean = %f std = %f\n", ratio, sqrt(variance));

    return true;
}

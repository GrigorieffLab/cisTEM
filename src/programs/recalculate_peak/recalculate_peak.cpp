#include "../../core/core_headers.h"

class
        RecalculatePeakApp : public MyApp {
  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

class TemplateComparisonObject {
  public:
    Image *          input_reconstruction, *windowed_particle, *projection_filter;
    AnglesAndShifts* angles;
    float            pixel_size_factor;
    int              search_mode;
    //	int							slice = 1;
};

// This is the function which will be minimized
Peak TemplateScore(void* scoring_parameters) {
    TemplateComparisonObject* comparison_object = reinterpret_cast<TemplateComparisonObject*>(scoring_parameters);
    Image                     current_projection;
    //	Peak box_peak;

    current_projection.Allocate(comparison_object->projection_filter->logical_x_dimension, comparison_object->projection_filter->logical_x_dimension, false);
    if ( comparison_object->input_reconstruction->logical_x_dimension != current_projection.logical_x_dimension ) {
        Image padded_projection;
        padded_projection.Allocate(comparison_object->input_reconstruction->logical_x_dimension, comparison_object->input_reconstruction->logical_x_dimension, false);
        comparison_object->input_reconstruction->ExtractSlice(padded_projection, *comparison_object->angles, 1.0f, false);
        padded_projection.SwapRealSpaceQuadrants( );
        padded_projection.BackwardFFT( );
        padded_projection.ChangePixelSize(&current_projection, comparison_object->pixel_size_factor, 0.001f, true);
        //		padded_projection.ChangePixelSize(&padded_projection, comparison_object->pixel_size_factor, 0.001f);
        //		padded_projection.ClipInto(&current_projection);
        //		current_projection.ForwardFFT();
    }
    else {
        comparison_object->input_reconstruction->ExtractSlice(current_projection, *comparison_object->angles, 1.0f, false);
        current_projection.SwapRealSpaceQuadrants( );
        current_projection.BackwardFFT( );
        current_projection.ChangePixelSize(&current_projection, comparison_object->pixel_size_factor, 0.001f, true);
    }

    //	current_projection.QuickAndDirtyWriteSlice("projections.mrc", comparison_object->slice);
    //	comparison_object->slice++;
    current_projection.MultiplyPixelWise(*comparison_object->projection_filter);
    //	current_projection.BackwardFFT();
    //	current_projection.AddConstant(-current_projection.ReturnAverageOfRealValuesOnEdges());
    //	current_projection.Resize(comparison_object->windowed_particle->logical_x_dimension, comparison_object->windowed_particle->logical_y_dimension, 1, 0.0f);
    //	current_projection.ForwardFFT();
    current_projection.ZeroCentralPixel( );
    current_projection.DivideByConstant(sqrtf(current_projection.ReturnSumOfSquares( )));
#ifdef MKL
    // Use the MKL
    vmcMulByConj(current_projection.real_memory_allocated / 2, reinterpret_cast<MKL_Complex8*>(comparison_object->windowed_particle->complex_values), reinterpret_cast<MKL_Complex8*>(current_projection.complex_values), reinterpret_cast<MKL_Complex8*>(current_projection.complex_values), VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
#else
    for ( long pixel_counter = 0; pixel_counter < current_projection.real_memory_allocated / 2; pixel_counter++ ) {
        current_projection.complex_values[pixel_counter] = std::conj(current_projection.complex_values[pixel_counter]) * comparison_object->windowed_particle->complex_values[pixel_counter];
    }
#endif
    current_projection.BackwardFFT( );
    //	wxPrintf("ping");
    if ( comparison_object->search_mode == 1 )
        return current_projection.FindPeakWithIntegerCoordinates( );
    else if ( comparison_object->search_mode == 2 ) {
        Peak current_peak;
        current_peak.x     = 0;
        current_peak.y     = 0;
        current_peak.value = current_projection.ReturnCentralPixelValue( );
        return current_peak;
    }

    //	box_peak = current_projection.FindPeakWithIntegerCoordinates();
    //	wxPrintf("address = %li\n", box_peak.physical_address_within_image);
    //	box_peak.x = 0.0f;
    //	box_peak.y = 0.0f;
    //	box_peak.value = current_projection.real_values[33152];
    //	return box_peak;
}

IMPLEMENT_APP(RecalculatePeakApp)

// override the DoInteractiveUserInput

void RecalculatePeakApp::DoInteractiveUserInput( ) {
    wxString input_search_images;
    wxString input_reconstruction;

    wxString mip_input_filename;
    wxString scaled_mip_input_filename;
    wxString best_psi_input_filename;
    wxString best_theta_input_filename;
    wxString best_phi_input_filename;
    wxString best_defocus_input_filename;
    wxString best_pixel_size_input_filename;
    wxString best_psi_output_file;
    wxString best_theta_output_file;
    wxString best_phi_output_file;
    wxString best_defocus_output_file;
    wxString best_pixel_size_output_file;

    wxString mip_output_file;
    wxString scaled_mip_output_file;

    float pixel_size              = 1.0f;
    float voltage_kV              = 300.0f;
    float spherical_aberration_mm = 2.7f;
    float amplitude_contrast      = 0.07f;
    float defocus1                = 10000.0f;
    float defocus2                = 10000.0f;

    float defocus_angle;
    float phase_shift;
    float low_resolution_limit    = 300.0f;
    float high_resolution_limit   = 8.0f;
    float angular_range           = 2.0f;
    float angular_step            = 5.0f;
    int   best_parameters_to_keep = 20;
    float defocus_search_range    = 1000;
    float defocus_search_step     = 10;
    //	float 		defocus_refine_step = 5;
    float pixel_size_search_range = 0.1f;
    float pixel_size_step         = 0.001f;
    //	float		pixel_size_refine_step = 0.001f;
    float    padding               = 1.0;
    bool     ctf_refinement        = false;
    float    mask_radius           = 0.0f;
    wxString my_symmetry           = "C1";
    float    in_plane_angular_step = 0;
    float    wanted_threshold;
    float    min_peak_radius;
    float    xy_change_threshold        = 10.0f;
    bool     exclude_above_xy_threshold = false;
    int      result_number              = 1;

    int max_threads;

    UserInput* my_input = new UserInput("RecalculatePeak", 1.00);

    input_search_images            = my_input->GetFilenameFromUser("Input images to be searched", "The input image stack, containing the images that should be searched", "image_stack.mrc", true);
    input_reconstruction           = my_input->GetFilenameFromUser("Input template reconstruction", "The 3D reconstruction from which projections are calculated", "reconstruction.mrc", true);
    mip_input_filename             = my_input->GetFilenameFromUser("Input MIP file", "The file with the maximum intensity projection image", "mip.mrc", false);
    scaled_mip_input_filename      = my_input->GetFilenameFromUser("Input scaled MIP file", "The file with the scaled MIP (peak search done on this image)", "scaled_mip.mrc", false);
    best_psi_input_filename        = my_input->GetFilenameFromUser("Input psi file", "The file with the best psi image", "psi.mrc", true);
    best_theta_input_filename      = my_input->GetFilenameFromUser("Input theta file", "The file with the best psi image", "theta.mrc", true);
    best_phi_input_filename        = my_input->GetFilenameFromUser("Input phi file", "The file with the best psi image", "phi.mrc", true);
    best_defocus_input_filename    = my_input->GetFilenameFromUser("Input defocus file", "The file with the best defocus image", "defocus.mrc", true);
    best_pixel_size_input_filename = my_input->GetFilenameFromUser("Input pixel size file", "The file with the best pixel size image", "pixel_size.mrc", true);

    wanted_threshold        = my_input->GetFloatFromUser("Peak threshold", "Peaks over this size will be taken", "7.5", 0.0);
    min_peak_radius         = my_input->GetFloatFromUser("Min peak radius (px.)", "Essentially the minimum closeness for peaks", "10.0", 0.0);
    pixel_size              = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
    voltage_kV              = my_input->GetFloatFromUser("Beam energy (keV)", "The energy of the electron beam used to image the sample in kilo electron volts", "300.0", 0.0);
    spherical_aberration_mm = my_input->GetFloatFromUser("Spherical aberration (mm)", "Spherical aberration of the objective lens in millimeters", "2.7");
    amplitude_contrast      = my_input->GetFloatFromUser("Amplitude contrast", "Assumed amplitude contrast", "0.07", 0.0, 1.0);
    defocus1                = my_input->GetFloatFromUser("Defocus1 (angstroms)", "Defocus1 for the input image", "10000", 0.0);
    defocus2                = my_input->GetFloatFromUser("Defocus2 (angstroms)", "Defocus2 for the input image", "10000", 0.0);
    defocus_angle           = my_input->GetFloatFromUser("Defocus angle (degrees)", "Defocus Angle for the input image", "0.0");
    phase_shift             = my_input->GetFloatFromUser("Phase shift (degrees)", "Additional phase shift in degrees", "0.0");
    //	low_resolution_limit = my_input->GetFloatFromUser("Low resolution limit (A)", "Low resolution limit of the data used for alignment in Angstroms", "300.0", 0.0);
    //	high_resolution_limit = my_input->GetFloatFromUser("High resolution limit (A)", "High resolution limit of the data used for alignment in Angstroms", "8.0", 0.0);
    //	angular_range = my_input->GetFloatFromUser("Angular refinement range", "AAngular range to refine", "2.0", 0.1);

    padding = my_input->GetFloatFromUser("Padding factor", "Factor determining how much the input volume is padded to improve projections", "2.0", 1.0);
    //	ctf_refinement = my_input->GetYesNoFromUser("Refine defocus", "Should the particle defocus be refined?", "No");
    mask_radius = my_input->GetFloatFromUser("Mask radius (A) (0.0 = no mask)", "Radius of a circular mask to be applied to the input particles during refinement", "0.0", 0.0);
    //	my_symmetry = my_input->GetSymmetryFromUser("Template symmetry", "The symmetry of the template reconstruction", "C1");
    xy_change_threshold        = my_input->GetFloatFromUser("Moved peak warning (A)", "Threshold for displaying warning of peak location changes during refinement", "10.0", 0.0);
    exclude_above_xy_threshold = my_input->GetYesNoFromUser("Exclude moving peaks", "Should the peaks that move more than the threshold be excluded from the output MIPs?", "No");
    result_number              = my_input->GetIntFromUser("Result number to refine", "If input files contain results from several searches, which one should be refined?", "1", 1);

#ifdef _OPENMP
    max_threads = my_input->GetIntFromUser("Max. threads to use for calculation", "When threading, what is the max threads to run", "1", 1);
#else
    max_threads = 1;
#endif
    float fixed_phi    = my_input->GetFloatFromUser("Constrained search Phi", "max phi for constrained search", "360.0", -360.0, 360.0);
    float fixed_theta  = my_input->GetFloatFromUser("Constrained search Theta", "max theta for constrained search", "360.0", -360.0, 360.0);
    float fixed_psi    = my_input->GetFloatFromUser("Constrained search Psi", "max psi for constrained search", "360.0", -360.0, 360.0);
    float coord_x      = my_input->GetFloatFromUser("Constrained search x", "coord x", "0.0", 0, 5120);
    float coord_y      = my_input->GetFloatFromUser("Constrained search y", "coord y", "0.0", 0, 5120);
    float d_pixel_size = my_input->GetFloatFromUser("change in pixel size", "d pixel size", "0.0", 0.0, 1.0);
    float d_defocus    = my_input->GetFloatFromUser("change in defocus", "d defocus", "0.0", -1200.0, 1200.0);

    int      first_search_position           = -1;
    int      last_search_position            = -1;
    int      image_number_for_gui            = 0;
    int      number_of_jobs_per_image_in_gui = 0;
    float    threshold_for_result_plotting   = 0.0f;
    wxString filename_for_gui_result_image;

    int search_mode = my_input->GetIntFromUser("search mode", "search mode", "1", 0, 3);

    wxString directory_for_results = "/dev/null"; // shouldn't be used in interactive

    delete my_input;

    //	my_current_job.Reset(42);
    my_current_job.ManualSetArguments("ttfffffffffiffftttttttfffbtiifffffffi", input_search_images.ToUTF8( ).data( ),
                                      input_reconstruction.ToUTF8( ).data( ),
                                      pixel_size,
                                      voltage_kV,
                                      spherical_aberration_mm,
                                      amplitude_contrast,
                                      defocus1,
                                      defocus2,
                                      defocus_angle,
                                      low_resolution_limit,
                                      high_resolution_limit,
                                      best_parameters_to_keep,
                                      padding,
                                      mask_radius,
                                      phase_shift,
                                      mip_input_filename.ToUTF8( ).data( ),
                                      scaled_mip_input_filename.ToUTF8( ).data( ),
                                      best_psi_input_filename.ToUTF8( ).data( ),
                                      best_theta_input_filename.ToUTF8( ).data( ),
                                      best_phi_input_filename.ToUTF8( ).data( ),
                                      best_defocus_input_filename.ToUTF8( ).data( ),
                                      best_pixel_size_input_filename.ToUTF8( ).data( ),
                                      wanted_threshold,
                                      min_peak_radius,
                                      xy_change_threshold,
                                      exclude_above_xy_threshold,
                                      my_symmetry.ToUTF8( ).data( ),
                                      result_number,
                                      max_threads,
                                      fixed_phi,
                                      fixed_theta,
                                      fixed_psi,
                                      coord_x,
                                      coord_y,
                                      d_pixel_size,
                                      d_defocus,
                                      search_mode);
}

// override the do calculation method which will be what is actually run..

bool RecalculatePeakApp::DoCalculation( ) {
    wxDateTime start_time = wxDateTime::Now( );

    wxString input_search_images_filename   = my_current_job.arguments[0].ReturnStringArgument( );
    wxString input_reconstruction_filename  = my_current_job.arguments[1].ReturnStringArgument( );
    float    pixel_size                     = my_current_job.arguments[2].ReturnFloatArgument( );
    float    voltage_kV                     = my_current_job.arguments[3].ReturnFloatArgument( );
    float    spherical_aberration_mm        = my_current_job.arguments[4].ReturnFloatArgument( );
    float    amplitude_contrast             = my_current_job.arguments[5].ReturnFloatArgument( );
    float    defocus1                       = my_current_job.arguments[6].ReturnFloatArgument( );
    float    defocus2                       = my_current_job.arguments[7].ReturnFloatArgument( );
    float    defocus_angle                  = my_current_job.arguments[8].ReturnFloatArgument( );
    float    low_resolution_limit           = my_current_job.arguments[9].ReturnFloatArgument( );
    float    high_resolution_limit_search   = my_current_job.arguments[10].ReturnFloatArgument( );
    int      best_parameters_to_keep        = my_current_job.arguments[11].ReturnIntegerArgument( );
    float    padding                        = my_current_job.arguments[12].ReturnFloatArgument( );
    float    mask_radius                    = my_current_job.arguments[13].ReturnFloatArgument( );
    float    phase_shift                    = my_current_job.arguments[14].ReturnFloatArgument( );
    wxString mip_input_filename             = my_current_job.arguments[15].ReturnStringArgument( );
    wxString scaled_mip_input_filename      = my_current_job.arguments[16].ReturnStringArgument( );
    wxString best_psi_input_filename        = my_current_job.arguments[17].ReturnStringArgument( );
    wxString best_theta_input_filename      = my_current_job.arguments[18].ReturnStringArgument( );
    wxString best_phi_input_filename        = my_current_job.arguments[19].ReturnStringArgument( );
    wxString best_defocus_input_filename    = my_current_job.arguments[20].ReturnStringArgument( );
    wxString best_pixel_size_input_filename = my_current_job.arguments[21].ReturnStringArgument( );
    float    wanted_threshold               = my_current_job.arguments[22].ReturnFloatArgument( );
    float    min_peak_radius                = my_current_job.arguments[23].ReturnFloatArgument( );
    float    xy_change_threshold            = my_current_job.arguments[24].ReturnFloatArgument( );
    bool     exclude_above_xy_threshold     = my_current_job.arguments[25].ReturnBoolArgument( );
    wxString my_symmetry                    = my_current_job.arguments[26].ReturnStringArgument( );
    int      result_number                  = my_current_job.arguments[27].ReturnIntegerArgument( );
    int      max_threads                    = my_current_job.arguments[28].ReturnIntegerArgument( );
    float    fixed_phi                      = my_current_job.arguments[29].ReturnFloatArgument( );
    float    fixed_theta                    = my_current_job.arguments[30].ReturnFloatArgument( );
    float    fixed_psi                      = my_current_job.arguments[31].ReturnFloatArgument( );
    float    coord_x                        = my_current_job.arguments[32].ReturnFloatArgument( );
    float    coord_y                        = my_current_job.arguments[33].ReturnFloatArgument( );
    float    d_pixel_size                   = my_current_job.arguments[34].ReturnFloatArgument( );
    float    d_defocus                      = my_current_job.arguments[35].ReturnFloatArgument( );
    float    search_mode                    = my_current_job.arguments[36].ReturnIntegerArgument( );

    int  i, j;
    bool parameter_map[5]; // needed for euler search init
    for ( i = 0; i < 5; i++ ) {
        parameter_map[i] = true;
    }

    float outer_mask_radius;

    float  temp_float;
    double temp_double_array[5];

    int   number_of_rotations;
    long  total_correlation_positions;
    long  current_correlation_position;
    long  pixel_counter;
    float sq_dist_x, sq_dist_y;
    long  address;
    long  best_address;

    int current_x;
    int current_y;

    AnglesAndShifts          angles;
    TemplateComparisonObject template_object;

    ImageFile input_search_image_file;
    ImageFile mip_input_file;
    ImageFile scaled_mip_input_file;
    ImageFile best_psi_input_file;
    ImageFile best_theta_input_file;
    ImageFile best_phi_input_file;
    ImageFile best_defocus_input_file;
    ImageFile best_pixel_size_input_file;
    ImageFile input_reconstruction_file;

    Curve whitening_filter;
    Curve number_of_terms;

    // mode 1: search for largest peak; mode 2: recalulate peak at original position;

    input_search_image_file.OpenFile(input_search_images_filename.ToStdString( ), false);
    mip_input_file.OpenFile(mip_input_filename.ToStdString( ), false);
    scaled_mip_input_file.OpenFile(scaled_mip_input_filename.ToStdString( ), false);
    best_psi_input_file.OpenFile(best_psi_input_filename.ToStdString( ), false);
    best_theta_input_file.OpenFile(best_theta_input_filename.ToStdString( ), false);
    best_phi_input_file.OpenFile(best_phi_input_filename.ToStdString( ), false);
    best_defocus_input_file.OpenFile(best_defocus_input_filename.ToStdString( ), false);
    best_pixel_size_input_file.OpenFile(best_pixel_size_input_filename.ToStdString( ), false);
    input_reconstruction_file.OpenFile(input_reconstruction_filename.ToStdString( ), false);

    Image input_image;
    Image windowed_particle;
    Image padded_reference;
    Image input_reconstruction;
    //	Image template_reconstruction;
    //	Image current_projection;
    //	Image padded_projection;

    Image projection_filter;

    Image mip_image;
    Image scaled_mip_image, scaled_mip_image_local;
    Image psi_image;
    Image theta_image;
    Image phi_image;
    Image defocus_image;
    Image pixel_size_image;
    Image best_psi, best_psi_local;
    Image best_theta, best_theta_local;
    Image best_phi, best_phi_local;
    Image best_defocus, best_defocus_local;
    Image best_pixel_size, best_pixel_size_local;
    Image best_mip, best_mip_local;
    Image best_scaled_mip, best_scaled_mip_local;

    Peak current_peak;
    Peak template_peak;
    Peak best_peak;
    long current_address;
    long address_offset;

    float current_phi;
    float current_theta;
    float current_psi;
    float current_defocus;
    float current_pixel_size;
    float best_score;
    float score;
    float starting_score;
    bool  first_score;

    float best_phi_score;
    float best_theta_score;
    float best_psi_score;
    float best_defocus_score;
    float best_pixel_size_score;
    int   ii, jj, kk, ll;
    float mult_i;
    float mult_i_start;
    float defocus_step;
    float score_adjustment;
    float offset_distance;
    //	float offset_warning_threshold = 10.0f;

    int   number_of_peaks_found = 0;
    int   peak_number;
    float mask_falloff     = 20.0;
    float min_peak_radius2 = powf(min_peak_radius, 2);

    if ( (input_search_image_file.ReturnZSize( ) < result_number) || (mip_input_file.ReturnZSize( ) < result_number) || (scaled_mip_input_file.ReturnZSize( ) < result_number) || (best_psi_input_file.ReturnZSize( ) < result_number) || (best_theta_input_file.ReturnZSize( ) < result_number) || (best_phi_input_file.ReturnZSize( ) < result_number) || (best_defocus_input_file.ReturnZSize( ) < result_number) || (best_pixel_size_input_file.ReturnZSize( ) < result_number) ) {
        SendErrorAndCrash("Error: Input files do not contain selected result\n");
    }
    input_image.ReadSlice(&input_search_image_file, result_number);
    mip_image.ReadSlice(&mip_input_file, result_number);
    scaled_mip_image.ReadSlice(&scaled_mip_input_file, result_number);
    psi_image.ReadSlice(&best_psi_input_file, result_number);
    theta_image.ReadSlice(&best_theta_input_file, result_number);
    phi_image.ReadSlice(&best_phi_input_file, result_number);
    defocus_image.ReadSlice(&best_defocus_input_file, result_number);
    pixel_size_image.ReadSlice(&best_pixel_size_input_file, result_number);
    padded_reference.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
    best_mip.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
    best_psi.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
    best_theta.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
    best_phi.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
    best_defocus.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
    best_pixel_size.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);

    best_psi.SetToConstant(0.0f);
    best_theta.SetToConstant(0.0f);
    best_phi.SetToConstant(0.0f);
    best_defocus.SetToConstant(0.0f);
    best_pixel_size.SetToConstant(0.0f);

    // Some settings for testing
    //	padding = 2.0f;
    //	ctf_refinement = true;
    //	defocus_search_range = 200.0f;
    //	defocus_step = 50.0f;

    input_reconstruction.ReadSlices(&input_reconstruction_file, 1, input_reconstruction_file.ReturnNumberOfSlices( ));
    if ( padding != 1.0f ) {
        input_reconstruction.Resize(input_reconstruction.logical_x_dimension * padding, input_reconstruction.logical_y_dimension * padding, input_reconstruction.logical_z_dimension * padding, input_reconstruction.ReturnAverageOfRealValuesOnEdges( ));
    }
    input_reconstruction.ForwardFFT( );
    //input_reconstruction.CosineMask(0.1, 0.01, true);
    //input_reconstruction.Whiten();
    //if (first_search_position == 0) input_reconstruction.QuickAndDirtyWriteSlices("/tmp/filter.mrc", 1, input_reconstruction.logical_z_dimension);
    input_reconstruction.ZeroCentralPixel( );
    input_reconstruction.SwapRealSpaceQuadrants( );

    CTF input_ctf;

    temp_float = (float(input_reconstruction_file.ReturnXSize( )) / 2.0f - 1.0f) * pixel_size;
    if ( mask_radius > temp_float )
        mask_radius = temp_float;

    // for now, I am assuming the MTF has been applied already.
    // work out the filter to just whiten the image..

    whitening_filter.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((input_image.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
    number_of_terms.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((input_image.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));

    wxDateTime my_time_out;
    wxDateTime my_time_in;

    // remove outliers

    input_image.ReplaceOutliersWithMean(5.0f);
    input_image.ForwardFFT( );

    input_image.ZeroCentralPixel( );
    input_image.Compute1DPowerSpectrumCurve(&whitening_filter, &number_of_terms);
    whitening_filter.SquareRoot( );
    whitening_filter.Reciprocal( );
    whitening_filter.MultiplyByConstant(1.0f / whitening_filter.ReturnMaximumValue( ));

    input_image.ApplyCurveFilter(&whitening_filter);
    input_image.ZeroCentralPixel( );
    input_image.DivideByConstant(sqrt(input_image.ReturnSumOfSquares( )));
    input_image.BackwardFFT( );

    //	long *addresses = new long[input_image.logical_x_dimension * input_image.logical_y_dimension / 100];

    // count total searches (lazy)

    best_scaled_mip.CopyFrom(&scaled_mip_image);

    //	best_mip.CopyFrom(&mip_image);
    //	best_scaled_mip.CopyFrom(&scaled_mip_image);
    best_mip.SetToConstant(0.0f);
    best_scaled_mip.SetToConstant(0.0f);
    best_psi.CopyFrom(&psi_image);
    best_theta.CopyFrom(&theta_image);
    best_phi.CopyFrom(&phi_image);
    best_defocus.CopyFrom(&defocus_image);
    best_pixel_size.CopyFrom(&pixel_size_image);

    ArrayOfTemplateMatchFoundPeakInfos all_peak_changes;
    ArrayOfTemplateMatchFoundPeakInfos all_peak_infos;

    TemplateMatchFoundPeakInfo temp_peak;
    all_peak_changes.Alloc(number_of_peaks_found);
    all_peak_changes.Add(temp_peak, number_of_peaks_found);

    all_peak_infos.Alloc(number_of_peaks_found);
    all_peak_infos.Add(temp_peak, number_of_peaks_found);

    if ( max_threads > number_of_peaks_found )
        max_threads = number_of_peaks_found;

    input_ctf.Init(voltage_kV, spherical_aberration_mm, amplitude_contrast, defocus1, defocus2, defocus_angle, 0.0, 0.0, 0.0, pixel_size, deg_2_rad(phase_shift));
    windowed_particle.Allocate(input_reconstruction_file.ReturnXSize( ), input_reconstruction_file.ReturnXSize( ), true);
    projection_filter.Allocate(input_reconstruction_file.ReturnXSize( ), input_reconstruction_file.ReturnXSize( ), false);

    current_peak.value = FLT_MAX;
    //	best_mip_local.CopyFrom(&mip_image);
    //	best_scaled_mip_local.CopyFrom(&scaled_mip_image);
    best_mip_local.Allocate(mip_image.logical_x_dimension, mip_image.logical_y_dimension, true);
    best_scaled_mip_local.Allocate(scaled_mip_image.logical_x_dimension, scaled_mip_image.logical_y_dimension, true);
    best_mip_local.SetToConstant(0.0f);
    best_scaled_mip_local.SetToConstant(0.0f);
    scaled_mip_image_local.CopyFrom(&scaled_mip_image);
    best_psi_local.CopyFrom(&psi_image);
    best_theta_local.CopyFrom(&theta_image);
    best_phi_local.CopyFrom(&phi_image);
    best_defocus_local.CopyFrom(&defocus_image);
    best_pixel_size_local.CopyFrom(&pixel_size_image);
    //	number_of_peaks_found = 0;

    template_object.input_reconstruction = &input_reconstruction;
    template_object.windowed_particle    = &windowed_particle;
    template_object.projection_filter    = &projection_filter;
    template_object.angles               = &angles;
    template_object.search_mode          = search_mode;

    //	while (current_peak.value >= wanted_threshold)

    // look for a peak..

    //		current_peak = scaled_mip_image.FindPeakWithIntegerCoordinates(0.0, FLT_MAX, input_reconstruction_file.ReturnXSize() / 2 + 1);
    //		if (current_peak.value < wanted_threshold) break;

    // ok we have peak..
    coord_x = coord_x - scaled_mip_image_local.physical_address_of_box_center_x;
    coord_y = coord_y - scaled_mip_image_local.physical_address_of_box_center_y;
    padded_reference.CopyFrom(&input_image);
    padded_reference.RealSpaceIntegerShift(coord_x, coord_y);
    padded_reference.ClipInto(&windowed_particle);
    if ( mask_radius > 0.0f )
        windowed_particle.CosineMask(mask_radius / pixel_size, mask_falloff / pixel_size);
    windowed_particle.ForwardFFT( );
    windowed_particle.SwapRealSpaceQuadrants( );
    //		windowed_particle.ZeroCentralPixel();
    //		windowed_particle.DivideByConstant(sqrtf(windowed_particle.ReturnSumOfSquares()));
    template_object.pixel_size_factor = 1.0f;
    first_score                       = false;

    //		number_of_peaks_found++;

    // get angles and mask out the local area so it won't be picked again..

    address = 0;

    coord_x = coord_x + scaled_mip_image_local.physical_address_of_box_center_x;
    coord_y = coord_y + scaled_mip_image_local.physical_address_of_box_center_y;

    wxPrintf("Peak @ %f, %f\n", coord_x, coord_y);

    for ( j = 0; j < scaled_mip_image_local.logical_y_dimension; j++ ) {
        sq_dist_y = float(pow(j - coord_y, 2));
        for ( i = 0; i < scaled_mip_image_local.logical_x_dimension; i++ ) {
            sq_dist_x = float(pow(i - coord_x, 2));

            // The square centered at the pixel
            //				if ( sq_dist_x + sq_dist_y <= min_peak_radius2 )
            //				{
            //					scaled_mip_image_local.real_values[address] = -FLT_MAX;
            //				}

            if ( sq_dist_x == 0.0f && sq_dist_y == 0.0f ) {
                current_address    = address;
                current_phi        = fixed_phi;
                current_theta      = fixed_theta;
                current_psi        = fixed_psi;
                current_defocus    = d_defocus;
                current_pixel_size = d_pixel_size;
                best_score         = -FLT_MAX;
                angles.Init(current_phi, current_theta, current_psi, 0.0, 0.0);
                wxPrintf("angles = %f %f %f \n", current_phi, current_theta, current_psi);

                input_ctf.SetDefocus((defocus1 + current_defocus) / pixel_size, (defocus2 + current_defocus) / pixel_size, deg_2_rad(defocus_angle));
                projection_filter.CalculateCTFImage(input_ctf);
                projection_filter.ApplyCurveFilter(&whitening_filter);

                //					input_image.ForwardFFT();
                //					template_object.windowed_particle = &input_image;

                //					input_reconstruction.RandomisePhases(pixel_size / 20.0f);
                template_object.pixel_size_factor = (pixel_size + current_pixel_size) / pixel_size;
                template_peak                     = TemplateScore(&template_object);
                //					starting_score = template_peak.value;
                wxPrintf("0 peak x, y, value = %g %g %g\n", template_peak.x, template_peak.y, template_peak.value);
                //					float s = 0.0f, a = 0.0f;
                //					for (int k = 0; k < 10; k++)
                //					{
                //						input_reconstruction.RandomisePhases(pixel_size / 20.0f);
                //						template_peak = TemplateScore(&template_object);
                //						wxPrintf("%i peak x, y, value = %g %g %g\n", k + 1, template_peak.x, template_peak.y, template_peak.value);
                //						s += powf(template_peak.value, 2);
                //						a += template_peak.value;
                //					}
                //					a /= 10;
                //					s /= 10;
                //					s = sqrtf(s - powf(a, 2));
                //					wxPrintf("noise, SNR = %g %g\n", s, fabsf(starting_score - a) / s);
                //					exit(0);
                //					starting_score = scaled_mip_image.real_values[address];

                score_adjustment = scaled_mip_image.real_values[current_address] / mip_image.real_values[current_address];
                //					score_adjustment = mip_image.real_values[address] / template_peak.value / sqrtf(template_object.windowed_particle->logical_x_dimension * template_object.windowed_particle->logical_y_dimension);
                float new_score      = template_peak.value * sqrtf(template_object.windowed_particle->logical_x_dimension * template_object.windowed_particle->logical_y_dimension);
                float adjusted_score = new_score * score_adjustment;
                wxPrintf("old, new, adjusted score = %g %g %g\n", scaled_mip_image.real_values[address], new_score, adjusted_score);
            }
            address++;
        }
        address += scaled_mip_image_local.padding_jump_value;
    }

    return true;
}
#include "../../core/core_headers.h"

#include "../../constants/constants.h"
#include <vector>
#include <algorithm>
#include <utility>
#include <cmath>
#include <numeric>
#include "../../include/Eigen/Dense"

using namespace std;
using namespace Eigen;

vector<float> CalculateProbit(const vector<float>& x) {

    // Step 1: Create a vector of pairs (value, original index)
    vector<pair<float, int>> indexedValues;
    for ( int i = 0; i < x.size( ); ++i ) {
        indexedValues.push_back(make_pair(x[i], i));
    }

    // Step 2: Sort the vector of pairs by the values
    sort(indexedValues.begin( ), indexedValues.end( ), [](const pair<float, int>& a, const pair<float, int>& b) {
        return a.first < b.first; // sort in ascending order
    });

    // Step 3: Create a vector to store the ranks
    vector<int> ranks(x.size( ));

    // Assign ranks based on sorted positions
    for ( int i = 0; i < indexedValues.size( ); ++i ) {
        ranks[indexedValues[i].second] = i + 1; // rank starts from 1
    }

    vector<float> probit_x;
    for ( int rank : ranks ) {
        float rank_tmp = (rank - 0.5) / x.size( );
        float probit   = sqrt(2.0) * cisTEM_erfinv(2 * rank_tmp - 1.0);
        probit_x.push_back(probit);
    }

    return probit_x;
}

float rho_g2(float x, float y, const Matrix2f& C) {
    float detC        = C.determinant( ); // Determinant of the matrix C
    float coefficient = 1 / (2 * M_PI) * sqrt(detC);
    float exponent    = -0.5 * (x * x * C(0, 0) + x * y * (C(0, 1) + C(1, 0)) + y * y * C(1, 1));
    return coefficient * exp(exponent);
}

vector<float> linspace(float start, float end, int num) {
    vector<float> linspaced;
    float         delta = (end - start) / (num - 1);
    for ( int i = 0; i < num; ++i ) {
        linspaced.push_back(start + delta * i);
    }
    return linspaced;
}

pair<MatrixXf, MatrixXf> CalculateAnisotropicGaussianForProbit(const vector<float>& pro_x1_, const vector<float>& pro_x2_) {
    float         pro_lim[2] = {-4.5, 4.5};
    int           n_pro_x    = 128;
    int           n_pro_y    = n_pro_x + 1;
    vector<float> pro_x_     = linspace(pro_lim[0], pro_lim[1], n_pro_x);
    vector<float> pro_y_     = linspace(pro_lim[0], pro_lim[1], n_pro_y);

    float pro_dx = (pro_x_.back( ) - pro_x_.front( )) / (pro_x_.size( ) - 1);
    float pro_dy = (pro_y_.back( ) - pro_y_.front( )) / (pro_y_.size( ) - 1);

    // mesh grid
    MatrixXf pro_x__(n_pro_y, n_pro_x);
    MatrixXf pro_y__(n_pro_y, n_pro_x);
    for ( int i = 0; i < n_pro_y; ++i ) {
        for ( int j = 0; j < n_pro_x; ++j ) {
            pro_x__(i, j) = pro_x_[j];
            pro_y__(i, j) = pro_y_[i];
        }
    }

    // [x1,x2]
    MatrixXf tmp(2, pro_x1_.size( ));
    for ( size_t i = 0; i < pro_x1_.size( ); ++i ) {
        tmp(0, i) = pro_x1_[i];
        tmp(1, i) = pro_x2_[i];
    }

    // calculate covariance matrix
    MatrixXf              Cinv__ = tmp * tmp.transpose( ) / pro_x1_.size( );
    EigenSolver<MatrixXf> es(Cinv__);
    MatrixXf              Dinv_  = es.eigenvalues( ).real( );
    MatrixXf              Uinv__ = es.eigenvectors( ).real( );

    return {Dinv_, Uinv__};
}

vector<float> Calculate1QPValue(const vector<float>& ag_x1_,
                                const vector<float>& ag_x2_,
                                const Vector2f&      Dinv_,
                                const Matrix2f&      Uinv__) {
    int           n_r = ag_x1_.size( );
    vector<float> p1q_equ_r_(n_r, 0.0);
    float         tmp_a = sqrt(Dinv_.maxCoeff( ));
    float         tmp_b = sqrt(Dinv_.minCoeff( ));

    Eigen::Matrix2f Uinv = Uinv__;
    if ( Dinv_[0] <= Dinv_[1] ) {
        Uinv.col(0) = Uinv__.col(1);
        Uinv.col(1) = Uinv__.col(0);
    }
    if ( Uinv(1, 0) < 0 && Uinv(0, 0) < 0 ) {
        Uinv(1, 0) *= -1;
        Uinv(0, 0) *= -1;
    }

    float tmp_w = atan2(Uinv(1, 0), Uinv(0, 0));
    // wxPrintf("Major axis = %f\n", tmp_w * 180 / M_PI);

    float tmp_gamma = atan2(1.0, 0.5 * sin(2 * tmp_w) * (tmp_b / tmp_a - tmp_a / tmp_b));

    for ( int i = 0; i < n_r; ++i ) {
        float ag_x1   = ag_x1_[i];
        float ag_x2   = ag_x2_[i];
        float p1q_equ = 1.0;
        if ( ag_x1 > 0.5 && ag_x2 > 0.5 ) {
            float tmp_x_0 = cos(tmp_w) * ag_x1 + sin(tmp_w) * ag_x2;
            float tmp_x_1 = -sin(tmp_w) * ag_x1 + cos(tmp_w) * ag_x2;
            float tmp_y_0 = tmp_x_0 / max(1e-12f, tmp_a);
            float tmp_y_1 = tmp_x_1 / max(1e-12f, tmp_b);
            float tmp_y_r = sqrt(tmp_y_0 * tmp_y_0 + tmp_y_1 * tmp_y_1);
            p1q_equ       = exp(-tmp_y_r * tmp_y_r / 2) * tmp_gamma / (2 * M_PI);
        }
        p1q_equ_r_[i] = p1q_equ;
    }

    vector<float> negative_log_p1q_equ_r_(n_r);
    for ( int i = 0; i < n_r; ++i ) {
        negative_log_p1q_equ_r_[i] = -log(p1q_equ_r_[i]);
    }

    return negative_log_p1q_equ_r_;
}

class
        MakeTemplateResultDev : public MyApp {
  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(MakeTemplateResultDev)

// override the DoInteractiveUserInput

void MakeTemplateResultDev::DoInteractiveUserInput( ) {

    wxString input_scaled_mip_filename;
    wxString input_mip_filename;
    wxString input_best_psi_filename;
    wxString input_best_theta_filename;
    wxString input_best_phi_filename;
    wxString input_best_defocus_filename;
    wxString input_best_pixel_size_filename;
    wxString xyz_coords_filename;

    int   min_peak_radius;
    float pixel_size;
    int   ignore_N_pixels_from_the_border = -1;

    UserInput* my_input = new UserInput("MakeTemplateResultDev", 1.00);

    input_scaled_mip_filename       = my_input->GetFilenameFromUser("Input scaled MIP file", "The file for saving the maximum intensity projection image", "mip.mrc", false);
    input_mip_filename              = my_input->GetFilenameFromUser("Input MIP file", "The file for saving the maximum intensity projection image", "mip.mrc", false);
    input_best_psi_filename         = my_input->GetFilenameFromUser("Input psi file", "The file containing the best psi image", "psi.mrc", false);
    input_best_theta_filename       = my_input->GetFilenameFromUser("Input theta file", "The file containing the best psi image", "theta.mrc", false);
    input_best_phi_filename         = my_input->GetFilenameFromUser("Input phi file", "The file containing the best psi image", "phi.mrc", false);
    input_best_defocus_filename     = my_input->GetFilenameFromUser("Input defocus file", "The file with the best defocus image", "defocus.mrc", true);
    input_best_pixel_size_filename  = my_input->GetFilenameFromUser("Input pixel size file", "The file with the best pixel size image", "pixel_size.mrc", true);
    xyz_coords_filename             = my_input->GetFilenameFromUser("Output x,y,z coordinate file", "The file for saving the x,y,z coordinates of the found targets", "coordinates.txt", false);
    min_peak_radius                 = my_input->GetIntFromUser("Min Peak Radius (px.)", "Essentially the minimum closeness for peaks", "10", 1);
    pixel_size                      = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
    ignore_N_pixels_from_the_border = my_input->GetIntFromUser("Ignore N pixels from the edge of the MIP", "Defaults to 1/2 the template dimension (-1)", "-1", -1);
    delete my_input;

    //	my_current_job.Reset(14);
    my_current_job.ManualSetArguments("ttttttttifi", input_scaled_mip_filename.ToUTF8( ).data( ),
                                      input_mip_filename.ToUTF8( ).data( ),
                                      input_best_psi_filename.ToUTF8( ).data( ),
                                      input_best_theta_filename.ToUTF8( ).data( ),
                                      input_best_phi_filename.ToUTF8( ).data( ),
                                      input_best_defocus_filename.ToUTF8( ).data( ),
                                      input_best_pixel_size_filename.ToUTF8( ).data( ),
                                      xyz_coords_filename.ToUTF8( ).data( ),
                                      min_peak_radius,
                                      pixel_size,
                                      ignore_N_pixels_from_the_border);
}

// override the do calculation method which will be what is actually run..

bool MakeTemplateResultDev::DoCalculation( ) {

    wxDateTime start_time = wxDateTime::Now( );

    wxString input_scaled_mip_filename       = my_current_job.arguments[0].ReturnStringArgument( );
    wxString input_mip_filename              = my_current_job.arguments[1].ReturnStringArgument( );
    wxString input_best_psi_filename         = my_current_job.arguments[2].ReturnStringArgument( );
    wxString input_best_theta_filename       = my_current_job.arguments[3].ReturnStringArgument( );
    wxString input_best_phi_filename         = my_current_job.arguments[4].ReturnStringArgument( );
    wxString input_best_defocus_filename     = my_current_job.arguments[5].ReturnStringArgument( );
    wxString input_best_pixel_size_filename  = my_current_job.arguments[6].ReturnStringArgument( );
    wxString xyz_coords_filename             = my_current_job.arguments[7].ReturnStringArgument( );
    int      min_peak_radius                 = my_current_job.arguments[8].ReturnIntegerArgument( );
    float    pixel_size                      = my_current_job.arguments[9].ReturnFloatArgument( );
    int      ignore_N_pixels_from_the_border = my_current_job.arguments[10].ReturnIntegerArgument( );

    Image scaled_mip_image;
    Image mip_image;
    Image psi_image;
    Image theta_image;
    Image phi_image;
    Image defocus_image;
    Image pixel_size_image;
    int   result_number = 1;
    Peak  current_peak;

    float current_phi;
    float current_theta;
    float current_psi;
    float current_defocus;
    float current_pixel_size;

    int   number_of_peaks_found = 0;
    float sq_dist_x, sq_dist_y;
    long  address;
    long  text_file_access_type;

    vector<float>                         coordinates;
    vector<tuple<float, int, int, float>> localMaxima; // z-score, Row, Column, SNR

    text_file_access_type = OPEN_TO_WRITE;
    NumericTextFile coordinate_file(xyz_coords_filename, text_file_access_type, 10);
    coordinate_file.WriteCommentLine("         Psi          Theta            Phi              X              Y              Z      PixelSize           z-score           SNR           pval");

    mip_image.QuickAndDirtyReadSlice(input_mip_filename.ToStdString( ), result_number);
    scaled_mip_image.QuickAndDirtyReadSlice(input_scaled_mip_filename.ToStdString( ), result_number);
    psi_image.QuickAndDirtyReadSlice(input_best_psi_filename.ToStdString( ), result_number);
    theta_image.QuickAndDirtyReadSlice(input_best_theta_filename.ToStdString( ), result_number);
    phi_image.QuickAndDirtyReadSlice(input_best_phi_filename.ToStdString( ), result_number);
    defocus_image.QuickAndDirtyReadSlice(input_best_defocus_filename.ToStdString( ), result_number);
    pixel_size_image.QuickAndDirtyReadSlice(input_best_pixel_size_filename.ToStdString( ), result_number);
    int mip_x_dimension = mip_image.logical_x_dimension;
    int mip_y_dimension = mip_image.logical_y_dimension;

    if ( ignore_N_pixels_from_the_border > 0 && (ignore_N_pixels_from_the_border > mip_image.logical_x_dimension / 2 || ignore_N_pixels_from_the_border > mip_image.logical_y_dimension / 2) ) {
        wxPrintf("You have entered %d for ignore_N_pixels_from_the_border, which is too large given image half dimesnsions of %d (X) and %d (Y)",
                 ignore_N_pixels_from_the_border, mip_x_dimension / 2, mip_y_dimension / 2);
        exit(-1);
    }

    wxPrintf("\n");

    for ( int i = ignore_N_pixels_from_the_border; i <= mip_x_dimension - ignore_N_pixels_from_the_border; i++ ) {
        for ( int j = ignore_N_pixels_from_the_border; j <= mip_y_dimension - ignore_N_pixels_from_the_border; j++ ) {

            float maxVal = scaled_mip_image.ReturnRealPixelFromPhysicalCoord(i, j, 0);

            bool isLocalMax = true;
            // Check the neighborhood
            for ( int ki = -min_peak_radius; ki <= min_peak_radius; ++ki ) {
                for ( int kj = -min_peak_radius; kj <= min_peak_radius; ++kj ) {
                    int ni = i + ki; // neighbor row index
                    int nj = j + kj; // neighbor column index
                    // Check boundaries and find maximum
                    if ( ni >= 0 && ni < mip_x_dimension && nj >= 0 && nj < mip_y_dimension ) {

                        if ( scaled_mip_image.ReturnRealPixelFromPhysicalCoord(ni, nj, 0) > maxVal ) {
                            maxVal     = scaled_mip_image.ReturnRealPixelFromPhysicalCoord(ni, nj, 0);
                            isLocalMax = false;
                        }
                    }
                }
            }

            if ( isLocalMax && maxVal == scaled_mip_image.ReturnRealPixelFromPhysicalCoord(i, j, 0) ) {
                localMaxima.emplace_back(maxVal, i, j, mip_image.ReturnRealPixelFromPhysicalCoord(i, j, 0));
                number_of_peaks_found++;
            }
        }
    }

    // Sort based on the local z-score maxima (descending order)
    sort(localMaxima.begin( ), localMaxima.end( ), [](const tuple<float, int, int, float>& a, const tuple<float, int, int, float>& b) {
        return get<0>(a) > get<0>(b); // Sorting by first element of tuple, which is the z-score
    });

    // calculate 2DTM p-value
    vector<float> zScores;
    vector<float> SNRs;

    // extract z-scores as vector
    for ( int i = 0; i < localMaxima.size( ); ++i ) {
        zScores.push_back(get<0>(localMaxima[i]));
        SNRs.push_back(get<3>(localMaxima[i]));
    }

    // calculate quantile transformed values (probit function)
    auto pro_zscores = CalculateProbit(zScores);
    auto pro_snrs    = CalculateProbit(SNRs);
    // estimate joint anisotropic gaussian from quantile transformed data
    auto [Dinv_, Uinv__] = CalculateAnisotropicGaussianForProbit(pro_zscores, pro_snrs);

    auto neg_log_p1q_ = Calculate1QPValue(pro_zscores, pro_snrs, Dinv_, Uinv__);

    int coord_x, coord_y;
    // write cisTEM star output
    cisTEMParameters output_params;
    output_params.parameters_to_write.SetActiveParameters(PSI | THETA | PHI | ORIGINAL_X_POSITION | ORIGINAL_Y_POSITION | DEFOCUS_1 | DEFOCUS_2 | DEFOCUS_ANGLE | SCORE | MICROSCOPE_VOLTAGE | MICROSCOPE_CS | AMPLITUDE_CONTRAST | BEAM_TILT_X | BEAM_TILT_Y | IMAGE_SHIFT_X | IMAGE_SHIFT_Y | ORIGINAL_IMAGE_FILENAME | PIXEL_SIZE);
    output_params.PreallocateMemoryAndBlank(localMaxima.size( ));

    for ( int rowId = 0; rowId < localMaxima.size( ); ++rowId ) {
        coordinates.clear( ); // efficient for reusing vector
        auto& current_peak       = localMaxima[rowId];
        coord_x                  = get<1>(current_peak);
        coord_y                  = get<2>(current_peak);
        float current_psi        = psi_image.ReturnRealPixelFromPhysicalCoord(coord_x, coord_y, 0);
        float current_theta      = theta_image.ReturnRealPixelFromPhysicalCoord(coord_x, coord_y, 0);
        float current_phi        = phi_image.ReturnRealPixelFromPhysicalCoord(coord_x, coord_y, 0);
        float current_defocus    = defocus_image.ReturnRealPixelFromPhysicalCoord(coord_x, coord_y, 0); // FIXME what to do with defocus
        float current_pixel_size = pixel_size_image.ReturnRealPixelFromPhysicalCoord(coord_x, coord_y, 0); // FIXME what to do with pixel size
        coordinates.push_back(current_psi);
        coordinates.push_back(current_theta);
        coordinates.push_back(current_phi);
        coordinates.push_back(coord_x * pixel_size);
        coordinates.push_back(coord_y * pixel_size);
        coordinates.push_back(current_defocus);
        coordinates.push_back(current_pixel_size);
        coordinates.push_back(get<0>(current_peak)); // z-score
        coordinates.push_back(get<3>(current_peak)); // SNR
        coordinates.push_back(neg_log_p1q_[rowId]); // neg log p-value

        // optional coordinate_file length
        coordinate_file.WriteLine(coordinates.data( ));

        //    output_params.parameters_to_write.SetActiveParameters(PSI | THETA | PHI | ORIGINAL_X_POSITION | ORIGINAL_Y_POSITION | DEFOCUS_1 | DEFOCUS_2 | DEFOCUS_ANGLE | SCORE | MICROSCOPE_VOLTAGE | MICROSCOPE_CS | AMPLITUDE_CONTRAST | BEAM_TILT_X | BEAM_TILT_Y | IMAGE_SHIFT_X | IMAGE_SHIFT_Y | ORIGINAL_IMAGE_FILENAME | PIXEL_SIZE);
        // FIXME: how to include defocus and image name into output star? How to append files from different images?
        // CHECK ME: How was template_matches_package.star created in the first place?
        // output cisTEM formatted star file
        output_params.all_parameters[rowId].position_in_stack                  = 1;
        output_params.all_parameters[rowId].psi                                = current_psi;
        output_params.all_parameters[rowId].theta                              = current_theta;
        output_params.all_parameters[rowId].phi                                = current_phi;
        output_params.all_parameters[rowId].original_x_position                = coord_x * pixel_size;
        output_params.all_parameters[rowId].original_y_position                = coord_y * pixel_size;
        output_params.all_parameters[rowId].defocus_1                          = 10000.0;
        output_params.all_parameters[rowId].defocus_2                          = 10000.0;
        output_params.all_parameters[rowId].defocus_angle                      = 20.0;
        output_params.all_parameters[rowId].score                              = get<0>(current_peak);
        output_params.all_parameters[rowId].microscope_voltage_kv              = 200.0;
        output_params.all_parameters[rowId].microscope_spherical_aberration_mm = 2.7;
        output_params.all_parameters[rowId].amplitude_contrast                 = 0.1;
        output_params.all_parameters[rowId].beam_tilt_x                        = 0.0;
        output_params.all_parameters[rowId].beam_tilt_y                        = 0.0;
        output_params.all_parameters[rowId].image_shift_x                      = 0.0;
        output_params.all_parameters[rowId].image_shift_y                      = 0.0;
        output_params.all_parameters[rowId].original_image_filename            = "test.mrc";
        output_params.all_parameters[rowId].pixel_size                         = 1.112;
        wxPrintf("one peak done\n");
    }

    output_params.WriteTocisTEMStarFile(wxString::Format("test.star").ToStdString( ), -1, -1, -1, -1);

    if ( is_running_locally == true ) {
        wxPrintf("\nFound %i peaks.\n\n", number_of_peaks_found);
        wxPrintf("\nMake Template Results: Normal termination\n");
        wxDateTime finish_time = wxDateTime::Now( );
        wxPrintf("Total Run Time : %s\n\n", finish_time.Subtract(start_time).Format("%Hh:%Mm:%Ss"));
    }

    return true;
}

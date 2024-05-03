#include "../../core/core_headers.h"

#include "../../constants/constants.h"
#include <vector>
#include <algorithm>
#include <utility>
#include <cmath>
#include <numeric>
#include "../../include/Eigen/Dense"
#include <thread>

using namespace std;
using namespace Eigen;

typedef struct image_asset {
    string filename;
    float  voltage;
    float  Cs;
    float  pixel_size;
} image_asset;

typedef struct tm_image_parameters {
    vector<int>    image_asset_id;
    vector<string> filename;
    vector<float>  voltage;
    vector<float>  Cs;
    vector<float>  pixel_size;
} tm_image_parameters;

optional<image_asset> getDataForAssetId(const tm_image_parameters& data, int assetId) {
    for ( size_t i = 0; i < data.image_asset_id.size( ); ++i ) {
        if ( data.image_asset_id[i] == assetId ) {
            image_asset result{
                    data.filename[i],
                    data.voltage[i],
                    data.Cs[i],
                    data.pixel_size[i]};
            return result;
        }
    }
    return nullopt; // Return an empty optional if not found
}

typedef struct tm_job {
    vector<int>      image_asset_id;
    vector<float>    defocus_1;
    vector<float>    defocus_2;
    vector<float>    defocus_angle;
    vector<float>    amplitude_contrast;
    vector<wxString> mip_filename;
    vector<wxString> scaled_mip_filename;
    vector<wxString> psi_filename;
    vector<wxString> theta_filename;
    vector<wxString> phi_filename;
} tm_job;

static int extract_image_parameters(void* data, int argc, char** argv, char** azColName) {
    tm_image_parameters* current_image = static_cast<tm_image_parameters*>(data); // Cast data back to the correct type

    for ( int i = 0; i < argc; i++ ) {
        string colName = azColName[i];
        if ( colName == "IMAGE_ASSET_ID" ) {
            // Assuming the id column contains integers
            current_image->image_asset_id.emplace_back(stoi(argv[i]));
        }
        else if ( colName == "FILENAME" ) {
            // Assuming the name column contains strings
            current_image->filename.emplace_back(argv[i]);
        }
        else if ( colName == "PIXEL_SIZE" ) {
            // Assuming the name column contains strings
            current_image->pixel_size.emplace_back(stof(argv[i]));
        }
        else if ( colName == "VOLTAGE" ) {
            // Assuming the name column contains strings
            current_image->voltage.emplace_back(stof(argv[i]));
        }
        else if ( colName == "SPHERICAL_ABERRATION" ) {
            // Assuming the name column contains strings
            current_image->Cs.emplace_back(stof(argv[i]));
        }
    }
    return 0;
}

static int extract_tm_parameters(void* data, int argc, char** argv, char** azColName) {
    tm_job* current_tm_job = static_cast<tm_job*>(data); // Cast data back to the correct type

    for ( int i = 0; i < argc; i++ ) {
        string colName = azColName[i];
        if ( colName == "IMAGE_ASSET_ID" ) {
            // Assuming the id column contains integers
            current_tm_job->image_asset_id.emplace_back(stoi(argv[i]));
        }
        else if ( colName == "USED_DEFOCUS1" ) {
            // Assuming the name column contains strings
            current_tm_job->defocus_1.emplace_back(stof(argv[i]));
        }
        else if ( colName == "USED_DEFOCUS2" ) {
            // Assuming the name column contains strings
            current_tm_job->defocus_2.emplace_back(stof(argv[i]));
        }
        else if ( colName == "USED_DEFOCUS_ANGLE" ) {
            // Assuming the name column contains strings
            current_tm_job->defocus_angle.emplace_back(stof(argv[i]));
        }
        else if ( colName == "USED_AMPLITUDE_CONTRAST" ) {
            // Assuming the name column contains strings
            current_tm_job->amplitude_contrast.emplace_back(stof(argv[i]));
        }
        else if ( colName == "MIP_OUTPUT_FILE" ) {
            // Assuming the name column contains strings
            current_tm_job->mip_filename.emplace_back(argv[i]);
        }
        else if ( colName == "SCALED_MIP_OUTPUT_FILE" ) {
            // Assuming the name column contains strings
            current_tm_job->scaled_mip_filename.emplace_back(argv[i]);
        }
        else if ( colName == "PSI_OUTPUT_FILE" ) {
            // Assuming the name column contains strings
            current_tm_job->psi_filename.emplace_back(argv[i]);
        }
        else if ( colName == "THETA_OUTPUT_FILE" ) {
            // Assuming the name column contains strings
            current_tm_job->theta_filename.emplace_back(argv[i]);
        }
        else if ( colName == "PHI_OUTPUT_FILE" ) {
            // Assuming the name column contains strings
            current_tm_job->phi_filename.emplace_back(argv[i]);
        }
    }
    return 0;
}

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
    wxString output_star_filename;

    int   min_peak_radius;
    float pixel_size;
    int   ignore_N_pixels_from_the_border = -1;

    bool     run_batch = false;
    wxString input_database_filename;
    int      tm_job_id      = 1;
    float    cutoff         = 7.0;
    int      sorting_metric = 1; // 1 for z-score, 2 for SNR, and 3 for p-value
    int      num_threads    = 1;

    UserInput* my_input = new UserInput("MakeTemplateResultDev", 1.00);
    run_batch           = my_input->GetYesNoFromUser("Run multiple images in a template match job?", "Individual image (false) or multiple images (true)", "No");
    if ( run_batch ) {
        input_database_filename = my_input->GetFilenameFromUser("Input database file", "The file for template match project", "tm.db", false);
        tm_job_id               = my_input->GetIntFromUser("Template Match Job ID", "template matching job id", "1", 1);
        output_star_filename    = my_input->GetFilenameFromUser("Output star file", "The star file containing the particle alignment parameters", "particle_stack.star", false);
        num_threads             = my_input->GetIntFromUser("Number of threads", "Max is number of images in TM job", "1", 1);
    }
    else {
        input_scaled_mip_filename      = my_input->GetFilenameFromUser("Input scaled MIP file", "The file for saving the maximum intensity projection image", "mip.mrc", false);
        input_mip_filename             = my_input->GetFilenameFromUser("Input MIP file", "The file for saving the maximum intensity projection image", "mip.mrc", false);
        input_best_psi_filename        = my_input->GetFilenameFromUser("Input psi file", "The file containing the best psi image", "psi.mrc", false);
        input_best_theta_filename      = my_input->GetFilenameFromUser("Input theta file", "The file containing the best psi image", "theta.mrc", false);
        input_best_phi_filename        = my_input->GetFilenameFromUser("Input phi file", "The file containing the best psi image", "phi.mrc", false);
        input_best_defocus_filename    = my_input->GetFilenameFromUser("Input defocus file", "The file with the best defocus image", "defocus.mrc", true);
        input_best_pixel_size_filename = my_input->GetFilenameFromUser("Input pixel size file", "The file with the best pixel size image", "pixel_size.mrc", true);
        xyz_coords_filename            = my_input->GetFilenameFromUser("Output x,y,z coordinate file", "The file for saving the x,y,z coordinates of the found targets", "coordinates.txt", false);
        pixel_size                     = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
    }

    min_peak_radius                 = my_input->GetIntFromUser("Min Peak Radius (px.)", "Essentially the minimum closeness for peaks", "10", 1);
    ignore_N_pixels_from_the_border = my_input->GetIntFromUser("Ignore N pixels from the edge of the MIP", "Default to 50px", "10", 0);
    sorting_metric                  = my_input->GetIntFromUser("Sorting metric -- z-score(1) SNR(2) p-value(3)", "Thresholding metric for 2DTM output", "1", 1, 3);
    cutoff                          = my_input->GetFloatFromUser("Sorting cutoff", "cutoff for selected metric", "7.5", 0.0);

    delete my_input;

    //	my_current_job.Reset(14);
    my_current_job.ManualSetArguments("btitittttttttfiiif", run_batch,
                                      input_database_filename.ToUTF8( ).data( ),
                                      tm_job_id,
                                      output_star_filename.ToUTF8( ).data( ),
                                      num_threads,
                                      input_scaled_mip_filename.ToUTF8( ).data( ),
                                      input_mip_filename.ToUTF8( ).data( ),
                                      input_best_psi_filename.ToUTF8( ).data( ),
                                      input_best_theta_filename.ToUTF8( ).data( ),
                                      input_best_phi_filename.ToUTF8( ).data( ),
                                      input_best_defocus_filename.ToUTF8( ).data( ),
                                      input_best_pixel_size_filename.ToUTF8( ).data( ),
                                      xyz_coords_filename.ToUTF8( ).data( ),
                                      pixel_size,
                                      min_peak_radius,
                                      ignore_N_pixels_from_the_border,
                                      sorting_metric,
                                      cutoff);
}

// override the do calculation method which will be what is actually run..

bool MakeTemplateResultDev::DoCalculation( ) {

    wxDateTime start_time                      = wxDateTime::Now( );
    bool       run_batch                       = my_current_job.arguments[0].ReturnBoolArgument( );
    wxString   input_database_filename         = my_current_job.arguments[1].ReturnStringArgument( );
    int        tm_job_id                       = my_current_job.arguments[2].ReturnIntegerArgument( );
    wxString   output_star_filename            = my_current_job.arguments[3].ReturnStringArgument( );
    int        num_threads                     = my_current_job.arguments[4].ReturnIntegerArgument( );
    wxString   input_scaled_mip_filename       = my_current_job.arguments[5].ReturnStringArgument( );
    wxString   input_mip_filename              = my_current_job.arguments[6].ReturnStringArgument( );
    wxString   input_best_psi_filename         = my_current_job.arguments[7].ReturnStringArgument( );
    wxString   input_best_theta_filename       = my_current_job.arguments[8].ReturnStringArgument( );
    wxString   input_best_phi_filename         = my_current_job.arguments[9].ReturnStringArgument( );
    wxString   input_best_defocus_filename     = my_current_job.arguments[10].ReturnStringArgument( );
    wxString   input_best_pixel_size_filename  = my_current_job.arguments[11].ReturnStringArgument( );
    wxString   xyz_coords_filename             = my_current_job.arguments[12].ReturnStringArgument( );
    float      pixel_size                      = my_current_job.arguments[13].ReturnFloatArgument( );
    int        min_peak_radius                 = my_current_job.arguments[14].ReturnIntegerArgument( );
    int        ignore_N_pixels_from_the_border = my_current_job.arguments[15].ReturnIntegerArgument( );
    int        sorting_metric                  = my_current_job.arguments[16].ReturnIntegerArgument( );
    float      cutoff                          = my_current_job.arguments[17].ReturnFloatArgument( );

    Image scaled_mip_image;
    Image mip_image;
    Image psi_image;
    Image theta_image;
    Image phi_image;
    Image defocus_image;
    Image pixel_size_image;

    float current_phi;
    float current_theta;
    float current_psi;
    float current_defocus;
    float current_pixel_size;

    int   number_of_peaks_found = 0;
    float sq_dist_x, sq_dist_y;
    long  address;
    long  text_file_access_type = OPEN_TO_WRITE;

    cisTEMParameterLine output_parameters;
    cisTEMParameters    output_star_file;
    // Preallocate space: number of peaks not known, so assume large enough number
    output_star_file.PreallocateMemoryAndBlank(1000000);
    output_star_file.parameters_to_write.SetActiveParameters(PSI | THETA | PHI | ORIGINAL_X_POSITION | ORIGINAL_Y_POSITION | DEFOCUS_1 | DEFOCUS_2 | DEFOCUS_ANGLE | SCORE | MICROSCOPE_VOLTAGE | MICROSCOPE_CS | AMPLITUDE_CONTRAST | BEAM_TILT_X | BEAM_TILT_Y | IMAGE_SHIFT_X | IMAGE_SHIFT_Y | ORIGINAL_IMAGE_FILENAME | PIXEL_SIZE);
    tm_image_parameters                   current_tm_images;
    tm_job                                current_tm_jobs;
    int                                   number_of_images = 1;
    vector<tuple<float, int, int, float>> localMaxima; // z-score, Row, Column, SNR
    vector<float>                         coordinates;

    if ( ! run_batch ) {
        // Read in images
        mip_image.QuickAndDirtyReadSlice(input_mip_filename.ToStdString( ), 1);
        scaled_mip_image.QuickAndDirtyReadSlice(input_scaled_mip_filename.ToStdString( ), 1);
        psi_image.QuickAndDirtyReadSlice(input_best_psi_filename.ToStdString( ), 1);
        theta_image.QuickAndDirtyReadSlice(input_best_theta_filename.ToStdString( ), 1);
        phi_image.QuickAndDirtyReadSlice(input_best_phi_filename.ToStdString( ), 1);
        defocus_image.QuickAndDirtyReadSlice(input_best_defocus_filename.ToStdString( ), 1);
        pixel_size_image.QuickAndDirtyReadSlice(input_best_pixel_size_filename.ToStdString( ), 1);

        if ( ignore_N_pixels_from_the_border > 0 && (ignore_N_pixels_from_the_border > mip_image.logical_x_dimension / 2 || ignore_N_pixels_from_the_border > mip_image.logical_y_dimension / 2) ) {
            wxPrintf("You have entered %d for ignore_N_pixels_from_the_border, which is too large given image half dimesnsions of %d (X) and %d (Y)",
                     ignore_N_pixels_from_the_border, mip_image.logical_x_dimension / 2, mip_image.logical_y_dimension / 2);
            exit(-1);
        }

        // Find local maxima in z-score map
        for ( int i = ignore_N_pixels_from_the_border; i <= mip_image.logical_x_dimension - ignore_N_pixels_from_the_border; i++ ) {
            for ( int j = ignore_N_pixels_from_the_border; j <= mip_image.logical_y_dimension - ignore_N_pixels_from_the_border; j++ ) {

                float maxVal = scaled_mip_image.ReturnRealPixelFromPhysicalCoord(i, j, 0);

                bool isLocalMax = true;
                // Check the neighborhood
                for ( int ki = -min_peak_radius; ki <= min_peak_radius; ++ki ) {
                    for ( int kj = -min_peak_radius; kj <= min_peak_radius; ++kj ) {
                        int ni = i + ki; // neighbor row index
                        int nj = j + kj; // neighbor column index
                        // Check boundaries and find maximum
                        if ( ni >= 0 && ni < mip_image.logical_x_dimension && nj >= 0 && nj < mip_image.logical_y_dimension ) {

                            if ( scaled_mip_image.ReturnRealPixelFromPhysicalCoord(ni, nj, 0) > maxVal ) {
                                maxVal     = scaled_mip_image.ReturnRealPixelFromPhysicalCoord(ni, nj, 0);
                                isLocalMax = false;
                            }
                        }
                    }
                }

                if ( isLocalMax && maxVal == scaled_mip_image.ReturnRealPixelFromPhysicalCoord(i, j, 0) ) {
                    localMaxima.emplace_back(maxVal, i, j, mip_image.ReturnRealPixelFromPhysicalCoord(i, j, 0));
                }
            }
        }
        wxPrintf("Found %i local maxima...\n", localMaxima.size( ));
        // Sort based on the local z-score maxima (descending order)
        sort(localMaxima.begin( ), localMaxima.end( ), [](const tuple<float, int, int, float>& a, const tuple<float, int, int, float>& b) {
            return get<0>(a) > get<0>(b); // Sorting by first element of tuple, which is the z-score
        });

        // Calculate 2DTM p-value
        vector<float> zScores;
        vector<float> SNRs;
        zScores.clear( );
        SNRs.clear( );

        // Extract z-scores and SNRs as vector
        for ( int i = 0; i < localMaxima.size( ); ++i ) {
            zScores.push_back(get<0>(localMaxima[i]));
            SNRs.push_back(get<3>(localMaxima[i]));
        }

        // Calculate quantile transformed values (probit function)
        auto pro_zscores = CalculateProbit(zScores);
        auto pro_snrs    = CalculateProbit(SNRs);
        // estimate joint anisotropic gaussian from quantile transformed data
        auto [Dinv_, Uinv__] = CalculateAnisotropicGaussianForProbit(pro_zscores, pro_snrs);

        // Calculate 2DTM p-values
        auto neg_log_p1q_ = Calculate1QPValue(pro_zscores, pro_snrs, Dinv_, Uinv__);

        vector<tuple<float, int, int, float, float>> filteredLocalMaxima;

        // Filter peaks based on specified metric and cutoff
        float sorting_val;
        for ( size_t i = 0; i < localMaxima.size( ); ++i ) {
            if ( sorting_metric == 1 ) {
                sorting_val = get<0>(localMaxima[i]); // z-score
            }
            else if ( sorting_metric == 2 ) {
                sorting_val = get<3>(localMaxima[i]); // SNR
            }
            else if ( sorting_metric == 3 ) {
                sorting_val = neg_log_p1q_[i];
            }
            if ( sorting_val >= cutoff ) { // Check if z-score is larger than a specified cutoff
                filteredLocalMaxima.emplace_back(make_tuple(
                        get<0>(localMaxima[i]), // z-score
                        get<1>(localMaxima[i]), // coord_x
                        get<2>(localMaxima[i]), // coord_y
                        get<3>(localMaxima[i]), // SNR
                        neg_log_p1q_[i] // p-value
                        ));
            }
        }
        wxPrintf("Found %i peaks above cutoff...\n", filteredLocalMaxima.size( ));
        number_of_peaks_found += filteredLocalMaxima.size( );

        // Write coordinate file with all three metrics
        NumericTextFile coordinate_file(xyz_coords_filename, text_file_access_type, 10);
        coordinate_file.WriteCommentLine("         Psi          Theta            Phi              X              Y              Z      PixelSize           z-score           SNR           pval");
        number_of_images = 1;
        for ( int rowId = 0; rowId < filteredLocalMaxima.size( ); ++rowId ) {

            coordinates.clear( ); // efficient for reusing vector
            auto& current_peak       = filteredLocalMaxima[rowId];
            int   coord_x            = get<1>(current_peak);
            int   coord_y            = get<2>(current_peak);
            float current_psi        = psi_image.ReturnRealPixelFromPhysicalCoord(coord_x, coord_y, 0);
            float current_theta      = theta_image.ReturnRealPixelFromPhysicalCoord(coord_x, coord_y, 0);
            float current_phi        = phi_image.ReturnRealPixelFromPhysicalCoord(coord_x, coord_y, 0);
            float current_defocus    = defocus_image.ReturnRealPixelFromPhysicalCoord(coord_x, coord_y, 0);
            float current_pixel_size = pixel_size_image.ReturnRealPixelFromPhysicalCoord(coord_x, coord_y, 0);
            coordinates.push_back(current_psi);
            coordinates.push_back(current_theta);
            coordinates.push_back(current_phi);
            coordinates.push_back(coord_x * pixel_size);
            coordinates.push_back(coord_y * pixel_size);
            coordinates.push_back(current_defocus);
            coordinates.push_back(current_pixel_size);
            coordinates.push_back(get<0>(current_peak)); // z-score
            coordinates.push_back(get<3>(current_peak)); // SNR
            coordinates.push_back(get<4>(current_peak)); // neg log p-value

            // optional coordinate_file length
            coordinate_file.WriteLine(coordinates.data( ));
        }
    }
    else {
        // test database
        sqlite3* db;
        char*    zErrMsg = 0;
        int      rc;

        // Open the database
        rc = sqlite3_open(input_database_filename, &db);
        if ( rc ) {
            wxPrintf("Can't open database: %s\n", sqlite3_errmsg(db));
            return 1;
        }
        else {
            wxPrintf("Opened database successfully\n");
        }
        // SQL to fetch data
        //const char* sql = "SELECT * FROM TEMPLATE_MATCH_LIST";
        string sql = "SELECT * FROM TEMPLATE_MATCH_LIST WHERE TEMPLATE_MATCH_JOB_ID = " + to_string(tm_job_id);

        // Execute SQL statement
        rc = sqlite3_exec(db, sql.c_str( ), extract_tm_parameters, &current_tm_jobs, &zErrMsg);
        if ( rc != SQLITE_OK ) {
            wxPrintf("SQL error\n");
            sqlite3_free(zErrMsg);
        }
        // debug printout
        //else {
        //    for ( size_t i = 0; i < current_tm_jobs.image_asset_id.size( ); i++ ) {
        //        wxPrintf("mip =%s\n", current_tm_jobs.mip_filename[i]);
        //    }
        //}

        sql = "SELECT * FROM IMAGE_ASSETS";

        // Execute SQL statement
        rc = sqlite3_exec(db, sql.c_str( ), extract_image_parameters, &current_tm_images, &zErrMsg);
        if ( rc != SQLITE_OK ) {
            wxPrintf("SQL error\n");
            sqlite3_free(zErrMsg);
        }
        // debug printout
        //else {
        //    for ( size_t i = 0; i < current_tm_images.image_asset_id.size( ); i++ ) {
        //        wxPrintf("image id = %i pixel size =%f\n", current_tm_images.image_asset_id[i], current_tm_images.pixel_size[i]);
        //    }
        //}

        // Close the database connection
        sqlite3_close(db);
        number_of_images = current_tm_jobs.image_asset_id.size( );
        num_threads      = min(min(num_threads, number_of_images), static_cast<int>(thread::hardware_concurrency( )));
        wxPrintf("Run %i image on %i threads...\n", number_of_images, num_threads);

        wxPrintf("\n");
        // Parallel when running multiple images in a TM job
#pragma omp parallel num_threads(num_threads) default(none) shared(current_tm_jobs, current_tm_images, ignore_N_pixels_from_the_border, min_peak_radius, output_star_file, number_of_peaks_found, number_of_images, cutoff, sorting_metric) private(localMaxima, mip_image, scaled_mip_image, psi_image, theta_image, phi_image, defocus_image, pixel_size_image, output_parameters)
        {
#pragma omp for schedule(dynamic, 1)
            // Loop through image
            for ( size_t img_idx = 0; img_idx < number_of_images; img_idx++ ) {
                wxPrintf("\n\n");
                // Clear peak vector for each image
                localMaxima.clear( );
                optional<image_asset> result;
                // Read in TM job parameters
                int image_asset_id = current_tm_jobs.image_asset_id[img_idx];
                result             = getDataForAssetId(current_tm_images, image_asset_id);
                wxPrintf("working on image ID = %i\n", image_asset_id);
                mip_image.QuickAndDirtyReadSlice(current_tm_jobs.mip_filename[img_idx].ToStdString( ), 1);
                scaled_mip_image.QuickAndDirtyReadSlice(current_tm_jobs.scaled_mip_filename[img_idx].ToStdString( ), 1);
                psi_image.QuickAndDirtyReadSlice(current_tm_jobs.psi_filename[img_idx].ToStdString( ), 1);
                theta_image.QuickAndDirtyReadSlice(current_tm_jobs.theta_filename[img_idx].ToStdString( ), 1);
                phi_image.QuickAndDirtyReadSlice(current_tm_jobs.phi_filename[img_idx].ToStdString( ), 1);

                if ( ignore_N_pixels_from_the_border > 0 && (ignore_N_pixels_from_the_border > mip_image.logical_x_dimension / 2 || ignore_N_pixels_from_the_border > mip_image.logical_y_dimension / 2) ) {
                    wxPrintf("You have entered %d for ignore_N_pixels_from_the_border, which is too large given image half dimesnsions of %d (X) and %d (Y)",
                             ignore_N_pixels_from_the_border, mip_image.logical_x_dimension / 2, mip_image.logical_y_dimension / 2);
                    exit(-1);
                }

                // Find local maxima in z-score map
                for ( int i = ignore_N_pixels_from_the_border; i <= mip_image.logical_x_dimension - ignore_N_pixels_from_the_border; i++ ) {
                    for ( int j = ignore_N_pixels_from_the_border; j <= mip_image.logical_y_dimension - ignore_N_pixels_from_the_border; j++ ) {

                        float maxVal = scaled_mip_image.ReturnRealPixelFromPhysicalCoord(i, j, 0);

                        bool isLocalMax = true;
                        // Check the neighborhood
                        for ( int ki = -min_peak_radius; ki <= min_peak_radius; ++ki ) {
                            for ( int kj = -min_peak_radius; kj <= min_peak_radius; ++kj ) {
                                int ni = i + ki; // neighbor row index
                                int nj = j + kj; // neighbor column index
                                // Check boundaries and find maximum
                                if ( ni >= 0 && ni < mip_image.logical_x_dimension && nj >= 0 && nj < mip_image.logical_y_dimension ) {

                                    if ( scaled_mip_image.ReturnRealPixelFromPhysicalCoord(ni, nj, 0) > maxVal ) {
                                        maxVal     = scaled_mip_image.ReturnRealPixelFromPhysicalCoord(ni, nj, 0);
                                        isLocalMax = false;
                                    }
                                }
                            }
                        }

                        if ( isLocalMax && maxVal == scaled_mip_image.ReturnRealPixelFromPhysicalCoord(i, j, 0) ) {
                            localMaxima.emplace_back(maxVal, i, j, mip_image.ReturnRealPixelFromPhysicalCoord(i, j, 0));
                        }
                    }
                }
                wxPrintf("Found %i local maxima...\n", localMaxima.size( ));
                // Sort based on the local z-score maxima (descending order)
                sort(localMaxima.begin( ), localMaxima.end( ), [](const tuple<float, int, int, float>& a, const tuple<float, int, int, float>& b) {
                    return get<0>(a) > get<0>(b);
                });

                // Extract z-scores as vector
                vector<float> zScores;
                vector<float> SNRs;
                zScores.clear( );
                SNRs.clear( );
                for ( int i = 0; i < localMaxima.size( ); ++i ) {
                    zScores.push_back(get<0>(localMaxima[i]));
                    SNRs.push_back(get<3>(localMaxima[i]));
                }

                // Calculate quantile transformed values (probit function)
                auto pro_zscores = CalculateProbit(zScores);
                auto pro_snrs    = CalculateProbit(SNRs);
                // Estimate joint anisotropic gaussian from quantile transformed data
                auto [Dinv_, Uinv__] = CalculateAnisotropicGaussianForProbit(pro_zscores, pro_snrs);

                auto neg_log_p1q_ = Calculate1QPValue(pro_zscores, pro_snrs, Dinv_, Uinv__);

                vector<tuple<float, int, int, float, float>> filteredLocalMaxima;

                // Filter and combine data
                float sorting_val;
                for ( size_t i = 0; i < localMaxima.size( ); ++i ) {
                    if ( sorting_metric == 1 ) {
                        sorting_val = get<0>(localMaxima[i]); // z-score
                    }
                    else if ( sorting_metric == 2 ) {
                        sorting_val = get<3>(localMaxima[i]); // SNR
                    }
                    else if ( sorting_metric == 3 ) {
                        sorting_val = neg_log_p1q_[i];
                    }
                    if ( sorting_val >= cutoff ) { // Check if z-score is larger than a specified cutoff
                        filteredLocalMaxima.emplace_back(make_tuple(
                                get<0>(localMaxima[i]), // z-score
                                get<1>(localMaxima[i]), // coord_x
                                get<2>(localMaxima[i]), // coord_y
                                get<3>(localMaxima[i]), // SNR
                                neg_log_p1q_[i] // p-value
                                ));
                    }
                }

                wxPrintf("Found %i peaks above cutoff...\n", filteredLocalMaxima.size( ));

                // write cisTEM star output for batch images
                for ( int rowId = 0; rowId < filteredLocalMaxima.size( ); ++rowId ) {
                    output_parameters.SetAllToZero( );
                    // output cisTEM formatted star file
                    auto& current_peak                                             = filteredLocalMaxima[rowId];
                    int   coord_x                                                  = get<1>(current_peak);
                    int   coord_y                                                  = get<2>(current_peak);
                    float current_psi                                              = psi_image.ReturnRealPixelFromPhysicalCoord(coord_x, coord_y, 0);
                    float current_theta                                            = theta_image.ReturnRealPixelFromPhysicalCoord(coord_x, coord_y, 0);
                    float current_phi                                              = phi_image.ReturnRealPixelFromPhysicalCoord(coord_x, coord_y, 0);
                    output_parameters.position_in_stack                            = 1;
                    output_parameters.psi                                          = current_psi;
                    output_parameters.theta                                        = current_theta;
                    output_parameters.phi                                          = current_phi;
                    output_parameters.original_x_position                          = coord_x * result->pixel_size;
                    output_parameters.original_y_position                          = coord_y * result->pixel_size;
                    output_parameters.defocus_1                                    = current_tm_jobs.defocus_1[img_idx];
                    output_parameters.defocus_2                                    = current_tm_jobs.defocus_2[img_idx];
                    output_parameters.defocus_angle                                = current_tm_jobs.defocus_angle[img_idx];
                    output_parameters.score                                        = get<0>(current_peak);
                    output_parameters.microscope_voltage_kv                        = result->voltage;
                    output_parameters.microscope_spherical_aberration_mm           = result->Cs;
                    output_parameters.amplitude_contrast                           = current_tm_jobs.amplitude_contrast[img_idx];
                    output_parameters.beam_tilt_x                                  = 0.0;
                    output_parameters.beam_tilt_y                                  = 0.0;
                    output_parameters.image_shift_x                                = 0.0;
                    output_parameters.image_shift_y                                = 0.0;
                    output_parameters.original_image_filename                      = result->filename;
                    output_parameters.pixel_size                                   = result->pixel_size;
                    output_star_file.all_parameters[rowId + number_of_peaks_found] = output_parameters;
                }
                wxPrintf("Written out peak at line %i - %i\n", number_of_peaks_found, filteredLocalMaxima.size( ) + number_of_peaks_found);

                number_of_peaks_found += filteredLocalMaxima.size( );
            } // End loop through images
        } // End openmp block
        output_star_file.WriteTocisTEMStarFile(output_star_filename, -1, -1, 1, -1);
    }

    if ( is_running_locally == true ) {
        wxPrintf("\nFound %i peaks.\n\n", number_of_peaks_found);
        wxPrintf("\nMake Template Results: Normal termination\n");
        wxDateTime finish_time = wxDateTime::Now( );
        wxPrintf("Total Run Time : %s\n\n", finish_time.Subtract(start_time).Format("%Hh:%Mm:%Ss"));
    }

    return true;
}

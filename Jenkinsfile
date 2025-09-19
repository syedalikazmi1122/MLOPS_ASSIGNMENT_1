pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/syedalikazmi1122/MLOPS_ASSIGNMENT_1.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    dockerImage = docker.build("kazmi11/mlops-assignment-1:${BUILD_NUMBER}")
                }
            }
        }

        stage('Push to DockerHub') {
            steps {
                script {
                    docker.withRegistry('https://index.docker.io/v1/', '5ca9f9f2-544d-406d-8d26-54187727283c') {
                        dockerImage.push()
                    }
                }
            }
        }
    }

    post {
        success {
            mail to: 'syedalikazmi0012@gmail.com',
                 subject: "Jenkins Job Successful",
                 body: "The Jenkins pipeline completed successfully and image has been pushed to DockerHub."
        }
        failure {
            mail to: 'syedalikazmi0012@gmail.com',
                 subject: "Jenkins Job Failed",
                 body: "The Jenkins pipeline failed. Please check Jenkins logs."
        }
    }
}
